"""Stellarmesh mesh.

name: mesh.py
author: Alex Koen, Sam Pasmann

desc: Mesh class wraps Gmsh functionality for geometry meshing.
"""

from __future__ import annotations

import logging
import math
import multiprocessing
import subprocess
import tempfile
import warnings
from contextlib import contextmanager
from multiprocessing import Process
from pathlib import Path
from typing import Optional

import gmsh
import numpy as np

from .geometry import Geometry

logger = logging.getLogger(__name__)

def _validate_process_and_thread_count(num_processes: int, num_threads: int):
    requested_thread_count = int(num_processes * num_threads)
    cpu_count = multiprocessing.cpu_count()
    if requested_thread_count > multiprocessing.cpu_count():
        msg = (f'Number of specified threads ({requested_thread_count}) '
                + f'exceeds CPU count ({cpu_count})')
        raise ValueError(msg)

# TODO(spasmann): if num_processes > num_solids, should raise Warning  # noqa: TD003
# and proceed with num_processes = num_solids
def _validate_process_and_solid_count(num_processes: int, geometry: Geometry):
    num_solids = len(geometry.solids)
    if num_processes > num_solids:
        msg = (f'Number of specified processes ({num_processes}) '
                + f'exceeds number of geometry solids ({num_solids})')
        raise ValueError(msg)

def _get_material_solid_map(geometry: Geometry):
    assert gmsh.is_initialized()

    material_solid_map = {}
    for s, m in zip(geometry.solids, geometry.material_names, strict=True):
        dim_tags = gmsh.model.occ.import_shapes_native_pointer(s._address())
        if dim_tags[0][0] != 3:
            raise TypeError("Importing non-solid geometry.")

        solid_tag = dim_tags[0][1]
        if m not in material_solid_map:
            material_solid_map[m] = [solid_tag]
        else:
            material_solid_map[m].append(solid_tag)

    return material_solid_map

def _mesh_geometry( # noqa: PLR0913
                process_id,
                geometry,
                mesh_data,
                min_mesh_size,
                max_mesh_size,
                curvature_mesh_size,
                dim,
                num_threads,
                scale_factor,):

    gmsh.initialize()
    gmsh.option.set_number("General.NumThreads", num_threads)
    gmsh.model.add(f"temp_model_{process_id}")
    material_solid_map = _get_material_solid_map(geometry)
    gmsh.model.occ.synchronize()

    # Scale volumes is scaling factor was specified
    if scale_factor is not None:
        logger.info(f"Scaling volumes by factor {scale_factor}")
        dim_tags = gmsh.model.getEntities(dim=3)
        gmsh.model.occ.dilate(
            dim_tags, 0.0, 0.0, 0.0, scale_factor, scale_factor, scale_factor
        )
        gmsh.model.occ.synchronize()

    for material, solid_tags in material_solid_map.items():
        gmsh.model.add_physical_group(3, solid_tags, name=f"mat:{material}")

    gmsh.option.set_number("Mesh.MeshSizeMin", min_mesh_size)
    gmsh.option.set_number("Mesh.MeshSizeMax", max_mesh_size)
    gmsh.option.set_number("Mesh.MeshSizeFromCurvature", curvature_mesh_size)
    gmsh.model.mesh.generate(dim)

    # Copy the mesh data
    m = {}
    for e in gmsh.model.getEntities():
        m[e] = (gmsh.model.getBoundary([e]),
                gmsh.model.mesh.getNodes(e[0], e[1]),
                gmsh.model.mesh.getElements(e[0], e[1]))
    mesh_data[process_id] = m

    gmsh.clear()
    gmsh.finalize()


class Mesh():
    """A Gmsh mesh.

    As gmsh allows for only a single process per thread, this class provides a context
    manager to set the gmsh api to operate on this mesh.
    """

    _mesh_filename: str

    def __init__(self, mesh_filename: Optional[str] = None):
        """Initialize a mesh from a .msh file.

        Args:
            mesh_filename: Optional .msh filename. If not provided defaults to a
            temporary file. Defaults to None.
        """
        if not mesh_filename:
            with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as mesh_file:
                mesh_filename = mesh_file.name
        self._mesh_filename = mesh_filename

    def __enter__(self):
        """Enter mesh context, setting gmsh commands to operate on this mesh."""
        if not gmsh.is_initialized():
            gmsh.initialize()

        gmsh.option.set_number(
            "General.Terminal",
            1 if logger.getEffectiveLevel() <= logging.INFO else 0,
        )
        gmsh.open(self._mesh_filename)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup (finalize) gmsh."""
        gmsh.finalize()

    def _save_changes(self, *, save_all: bool = True):
        gmsh.option.set_number("Mesh.SaveAll", 1 if save_all else 0)
        gmsh.write(self._mesh_filename)

    def write(self, filename: str, *, save_all: bool = True):
        """Write mesh to a .msh file.

        Args:
            filename: Path to write file.
            save_all: Whether to save all entities (or just physical groups). See
            Gmsh documentation for Mesh.SaveAll. Defaults to True.
        """
        with self:
            gmsh.option.set_number("Mesh.SaveAll", 1 if save_all else 0)
            gmsh.write(filename)

    @classmethod
    def from_geometry(  # noqa: PLR0913
        cls,
        geometry: Geometry,
        min_mesh_size: float = 50,
        max_mesh_size: float = 50,
        curvature_mesh_size: int = 0,
        dim: int = 2,
        *,
        num_threads: Optional[int] = None,
        num_processes: Optional[int] = 1,
        scale_factor: Optional[float] = None,
    ) -> Mesh:
        """Mesh solids with Gmsh.

        See Gmsh documentation on mesh sizes:
        https://gmsh.info/doc/texinfo/gmsh.html#Specifying-mesh-element-sizes

        Args:
            geometry: Geometry to be meshed.
            min_mesh_size: Min mesh element size. Defaults to 50.
            max_mesh_size: Max mesh element size. Defaults to 50.
            curvature_mesh_size: If set to a positive value, the mesh will be
            adapted with respect to the curvature of the model entities. The value
            giving the target number of elements per 2 Pi radians. Defaults to 0.
            dim: Generate a mesh up to this dimension. Defaults to 2.
            num_threads: Max number of threads to use when Gmsh compiled with OpenMP
            support. 0 for system default i.e. OMP_NUM_THREADS. Defaults to None.
            num_processes: Use Python's native multiprocess to launch several gmsh
            processes for potentially faster meshing time. Defaults to 1.
            scale_factor: Scaling factor for geometry. Defaults to None.
        """
        _validate_process_and_thread_count(num_processes, num_threads)

        _validate_process_and_solid_count(num_processes, geometry)

        logger.info("Meshing Solids With:\n"
                     +f"    Min Mesh Size: {min_mesh_size}\n"
                     +f"    Max Mesh Size: {max_mesh_size}\n"
                     +f"    Curvature Mesh Size: {curvature_mesh_size}\n"
                     +f"    Number of OMP Threads: {num_threads}\n"
                     +f"    Number of Processes: {num_processes}\n")

        #####################################################################
        # DISTRIBUTE WORK
        #####################################################################
        # This data structure is where independent processes will store mesh data for
        # final compilation at the end.
        manager = multiprocessing.Manager()
        mesh_data = manager.dict()

        # This bit of code distributes the number of geometry solids between the number
        # of specified processes.
        n_work = math.floor(len(geometry.solids) / num_processes)
        remainder = 1 if len(geometry.solids) % num_processes else 0
        procs = []
        for i in range(num_processes):
            work_start = int(n_work * i)
            work_end = int(n_work * (i+1))
            if i == num_processes-1:
                work_end += remainder
            # define new geometry for process i
            p_solids = geometry.solids[work_start:work_end]
            p_mat_names = geometry.material_names[work_start:work_end]
            p_geometry = Geometry(p_solids, material_names=p_mat_names)

            p = Process(target=_mesh_geometry,
                        args=(
                            i,
                            p_geometry,
                            mesh_data,
                            min_mesh_size,
                            max_mesh_size,
                            curvature_mesh_size,
                            dim,
                            num_threads,
                            scale_factor,))
            p.start()
            procs.append(p)
        # wait for processes to complete
        for p in procs:
            p.join()

        #####################################################################
        # COMBINE MESHES
        #####################################################################
        # This section of code writes all of the mesh data (nodes and elements) from the
        # individually meshed entities into one final model.
        # There was no 'merge mesh' function or the like sthat I could find. -spasmann
        with cls() as mesh:
        # gmsh.initialize()
            gmsh.model.add("stellarmesh_model")
            node_tag_count = 1
            elem_tags_map = {}

            for mesh_id in sorted(mesh_data):
                # print(f'\nMESHING DATA FROM MESH {mesh}\n')
                m = mesh_data[mesh_id]
                elem_tags_map[mesh_id] = {}
                element_tag_count = node_tag_count
                for e in m:
                    e_dim = e[0]
                    tag = e[1]
                    boundary = [b[1] for b in m[e][0]]
                    nodeTags = m[e][1][0]
                    nodeCoords = m[e][1][1]
                    elemTypes = m[e][2][0]
                    elemTags = m[e][2][1]
                    elemNodeTags = m[e][2][2]

                    # increment nodeTag
                    if nodeTags.size:
                        nodeTags = np.arange(node_tag_count,
                                            node_tag_count+nodeTags.size,
                                            dtype=np.uint64)
                        node_tag_count += nodeTags.size

                    # increment elementTags and elemNodeTags
                    if len(elemTags):
                        for i in range(len(elemTags)):
                            # elemTags is a unique and strictly positive set of tags.
                            # These tags will be used to map the element nodes
                            old_tags = elemTags[i]
                            new_tags = np.arange(element_tag_count,
                                                element_tag_count+old_tags.size,
                                                dtype=np.uint64)
                            element_tag_count += old_tags.size
                            elemTags[i] = new_tags
                            elem_tags_map[mesh_id].update({k:v for k,v in zip(old_tags,new_tags, strict=False)})
                            # elemNodeTags is a vector of equivalent length of elemTags,
                            # each entry is a vector of length equal to the number of
                            # elements of the given type times the number of N nodes this
                            # type of element.

                            # This line maps the original element node tags to the new set.
                            elemNodeTags[i] = np.array([elem_tags_map[mesh_id][v] for v in elemNodeTags[i]])

                    tag = gmsh.model.addDiscreteEntity(e_dim, -1, boundary)
                    gmsh.model.mesh.addNodes(e_dim, tag, nodeTags, nodeCoords)
                    gmsh.model.mesh.addElements(e_dim, tag, elemTypes, elemTags, elemNodeTags)

            gmsh.model.mesh.generate(dim)
            # mesh._save_changes()
            gmsh.write('mesh.msh')
        # gmsh.write(self.)
        # gmsh.clear()
        # gmsh.finalize()

    @classmethod
    def mesh_geometry(
        cls,
        geometry: Geometry,
        min_mesh_size: float = 50,
        max_mesh_size: float = 50,
        dim: int = 2,
    ) -> Mesh:
        """Mesh solids with Gmsh.

        See Gmsh documentation on mesh sizes:
        https://gmsh.info/doc/texinfo/gmsh.html#Specifying-mesh-element-sizes

        Args:
            geometry: Geometry to be meshed.
            min_mesh_size: Min mesh element size. Defaults to 50.
            max_mesh_size: Max mesh element size. Defaults to 50.
            dim: Generate a mesh up to this dimension. Defaults to 2.
        """
        warnings.warn(
            "The mesh_geometry method is deprecated. Use from_geometry instead.",
            FutureWarning,
            stacklevel=2,
        )
        return cls.from_geometry(geometry, min_mesh_size, max_mesh_size, dim)

    def render(
        self,
        output_filename: Optional[str] = None,
        rotation_xyz: tuple[float, float, float] = (0, 0, 0),
        normals: int = 0,
        *,
        clipping: bool = True,
    ) -> str:
        """Render mesh as an image.

        Args:
            output_filename: Optional output filename. Defaults to None.
            rotation_xyz: Rotation in Euler angles. Defaults to (0, 0, 0).
            normals: Normal render size. Defaults to 0.
            clipping: Whether to enable mesh clipping. Defaults to True.

        Returns:
            Path to image file, either passed output_filename or a temporary file.
        """
        with self:
            gmsh.option.set_number("Mesh.SurfaceFaces", 1)
            gmsh.option.set_number("Mesh.Clip", 1 if clipping else 0)
            gmsh.option.set_number("Mesh.Normals", normals)
            gmsh.option.set_number("General.Trackball", 0)
            gmsh.option.set_number("General.RotationX", rotation_xyz[0])
            gmsh.option.set_number("General.RotationY", rotation_xyz[1])
            gmsh.option.set_number("General.RotationZ", rotation_xyz[2])
            if not output_filename:
                with tempfile.NamedTemporaryFile(
                    delete=False, mode="w", suffix=".png"
                ) as temp_file:
                    output_filename = temp_file.name

            try:
                gmsh.fltk.initialize()
                gmsh.write(output_filename)
            finally:
                gmsh.fltk.finalize()
            return output_filename

    @staticmethod
    def _check_is_initialized():
        if not gmsh.is_initialized():
            raise RuntimeError("Gmsh not initialized.")

    @contextmanager
    def _stash_physical_groups(self):
        self._check_is_initialized()
        physical_groups: dict[tuple[int, int], tuple[list[int], str]] = {}
        dim_tags = gmsh.model.get_physical_groups()
        for dim_tag in dim_tags:
            tags = gmsh.model.get_entities_for_physical_group(*dim_tag)
            name = gmsh.model.get_physical_name(*dim_tag)
            physical_groups[dim_tag] = (tags, name)
        gmsh.model.remove_physical_groups(dim_tags)

        try:
            yield
        except Exception as e:
            raise RuntimeError("Cannot unstash physical groups due to error.") from e
        else:
            if len(gmsh.model.get_physical_groups()) > 0:
                raise RuntimeError(
                    "Not overwriting existing physical groups on stash restore."
                )
            for physical_group in physical_groups.items():
                dim, tag = physical_group[0]
                tags, name = physical_group[1]
                gmsh.model.add_physical_group(dim, tags, tag, name)

    def refine(  # noqa: PLR0913
        self,
        *,
        min_mesh_size: Optional[float] = None,
        max_mesh_size: Optional[float] = None,
        const_mesh_size: Optional[float] = None,
        hausdorff_value: float = 0.01,
        gradation_value: float = 1.3,
        optim: bool = False,
    ) -> Mesh:
        """Refine mesh using mmgs.

        See mmgs documentation:
        https://www.mmgtools.org/mmg-remesher-try-mmg/mmg-remesher-tutorials/mmg-remesher-mmg2d/mesh-adaptation-to-a-solution
        for more info.

        Pay particular attention to the hausdorff value, which overrides most of the
        other options and is typically set too low. Set to a large value, on the order
        of the size of your bounding box, to disable completely.

        Args:
            min_mesh_size: -hmin: Min size of output mesh elements. Defaults to None.
            max_mesh_size: -hmax: Max size of output mesh elements. Defaults to None.
            const_mesh_size: -hsize: Constant size map
            hausdorff_value: -hausd: Hausdorff value. Defaults to 0.01, which is
            suitable for a circle of radius 1. Set to a large value to disable effect.
            gradation_value: -hgrad Gradation value. Defaults to 1.3.
            optim: -optim Do not change elements sizes. Defaults to False.

        Raises:
            RuntimeError: If refinement fails.

        Returns:
            New refined mesh with filename <original-filename>.refined.msh.
        """
        with (
            self,
            self._stash_physical_groups(),
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            filename = f"{tmpdir}/model.mesh"
            gmsh.write(filename)

            refined_filename = str(Path(filename).with_suffix(".o.mesh").resolve())
            command = ["mmgs"]

            params = {
                "-hmin": min_mesh_size,
                "-hmax": max_mesh_size,
                "-hsiz": const_mesh_size,
                "-hausd": hausdorff_value,
                "-hgrad": gradation_value,
            }

            for param in params.items():
                if param[1]:
                    command.extend([param[0], str(param[1])])
            if optim:
                command.append("-optim")

            command.extend(
                [
                    "-in",
                    filename,
                    "-out",
                    refined_filename,
                ]
            )

            # TODO(akoen): log subprocess realtime
            # https://github.com/Thea-Energy/stellarmesh/issues/13
            try:
                logger.info(
                    f"Refining mesh {filename} with mmgs, output to {refined_filename}."
                )
                output = subprocess.run(
                    command,
                    text=True,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )

                if output.stdout:
                    logger.info(output.stdout)

            except subprocess.CalledProcessError as e:
                logger.exception(
                    "Command failed with error code %d\nSTDERR:%s",
                    e.returncode,
                    e.stdout,
                )
                raise RuntimeError("Command failed to run. See output above.") from e

            gmsh.model.mesh.clear()
            gmsh.merge(refined_filename)

            new_filename = str(
                Path(self._mesh_filename).with_suffix(".refined.msh").resolve()
            )
            gmsh.option.set_number("Mesh.SaveAll", 1)
            gmsh.write(new_filename)
            return type(self)(new_filename)
