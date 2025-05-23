import sys
from enum import Enum
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from pxr import Usd, UsdGeom

# CSV field names from RenderDoc.
# Leading white space is intentional to match the CSV output.
VTX = 'VTX'
IDX = ' IDX'
POSITION_X = ' POSITION.x'
POSITION_Y = ' POSITION.y'
POSITION_Z = ' POSITION.z'
POSITION_W = ' POSITION.w'
SV_POSITION_X = ' SV_Position.x'
SV_POSITION_Y = ' SV_Position.y'
SV_POSITION_Z = ' SV_Position.z'
SV_POSITION_W = ' SV_Position.w'
TEXCOORD0_X = ' TEXCOORD0.x'
TEXCOORD0_Y = ' TEXCOORD0.y'
TEXCOORD0_Z = ' TEXCOORD0.z'
TEXCOORD0_W = ' TEXCOORD0.w'
TEXCOORD1_X = ' TEXCOORD1.x'
TEXCOORD1_Y = ' TEXCOORD1.y'
TEXCOORD1_Z = ' TEXCOORD1.z'
TEXCOORD1_W = ' TEXCOORD1.w'
TEXCOORD2_X = ' TEXCOORD2.x'
TEXCOORD2_Y = ' TEXCOORD2.y'
TEXCOORD2_Z = ' TEXCOORD2.z'
TEXCOORD2_W = ' TEXCOORD2.w'
TEXCOORD3_X = ' TEXCOORD3.x'
TEXCOORD3_Y = ' TEXCOORD3.y'
TEXCOORD3_Z = ' TEXCOORD3.z'
TEXCOORD3_W = ' TEXCOORD3.w'
TEXCOORD4_X = ' TEXCOORD4.x'
TEXCOORD4_Y = ' TEXCOORD4.y'
TEXCOORD4_Z = ' TEXCOORD4.z'
TEXCOORD4_W = ' TEXCOORD4.w'

SNAKE3_WHEELS_FIELDS = [
    VTX, IDX,
    POSITION_X, POSITION_Y, POSITION_Z,
    TEXCOORD1_X, TEXCOORD1_Y,
    TEXCOORD2_X, TEXCOORD2_Y,
    TEXCOORD3_X, TEXCOORD3_Y,
    TEXCOORD4_X, TEXCOORD4_Y
]

SNAKE3_BODY_FIELDS = [
    VTX, IDX,
    POSITION_X, POSITION_Y, POSITION_Z, POSITION_W,
    TEXCOORD0_X, TEXCOORD0_Y, TEXCOORD0_Z, TEXCOORD0_W,
    TEXCOORD1_X, TEXCOORD1_Y,
    TEXCOORD2_X, TEXCOORD2_Y,
    TEXCOORD3_X, TEXCOORD3_Y, TEXCOORD3_Z, TEXCOORD3_W,
    TEXCOORD4_X, TEXCOORD4_Y
]

class MeshType(Enum):
    NONE = -1
    SNAKE3_WHEELS = 0
    SNAKE3_BODY = 1

class MeshData:
    def __init__(self):
        self.name: str = ''
        self.vertices: Optional[np.ndarray] = None
        self.texcoords0: Optional[np.ndarray] = None
        self.texcoords1: Optional[np.ndarray] = None
        self.normals: Optional[np.ndarray] = None
        self.indices: Optional[np.ndarray] = None

def csv_to_usd(csv_path: Path) -> None:
    data = pd.read_csv(csv_path)

    mesh_type = _get_mesh_type(data)
    if mesh_type == MeshType.SNAKE3_WHEELS:
        _process_snake3_wheels(csv_path, data)
    elif mesh_type == MeshType.SNAKE3_BODY:
        _process_snake3_body(csv_path, data)
    else:
        print('Unknown mesh type. Exiting.')

def _get_mesh_type(data: pd.DataFrame) -> MeshType:
    if set(data.columns) == set(SNAKE3_WHEELS_FIELDS):
        return MeshType.SNAKE3_WHEELS
    elif set(data.columns) == set(SNAKE3_BODY_FIELDS):
        return MeshType.SNAKE3_BODY

    return MeshType.NONE

def _save_usd(csv_path: Path, meshDataList: List[MeshData]) -> None:
    """
    Crate a USD file with the given meshes.
    Normals and texture coordinates are optional.
    Assumes all meshes are triangulated.
    """
    usd_path = csv_path.with_suffix('.usd')
    stage = Usd.Stage.CreateNew(usd_path.as_posix())
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

    for meshData in meshDataList:
        if not _is_mesh_valid(meshData):
            print(f"Invalid mesh data for {meshData.name}. Skipping.")
            continue

        mesh_path = f'/{csv_path.stem}_{meshData.name}'
        mesh = UsdGeom.Mesh.Define(stage, mesh_path)

        mesh.CreatePointsAttr(meshData.vertices)

        if meshData.normals is not None:
            mesh.CreateNormalsAttr(meshData.normals)

        if meshData.texcoords0 is not None:
            mesh.CreatePrimvar('st', meshData.texcoords0, UsdGeom.Tokens.textureCoordinate)

        if meshData.texcoords1 is not None:
            mesh.CreatePrimvar('st1', meshData.texcoords1, UsdGeom.Tokens.textureCoordinate)

        mesh.CreateFaceVertexCountsAttr([3] * (len(meshData.indices) // 3))
        mesh.CreateFaceVertexIndicesAttr(meshData.indices)

    stage.Save()
    print(f"USD file created: {usd_path}")

def _is_mesh_valid(mesh: MeshData) -> bool:
    if mesh.name == '':
        return False

    if mesh.vertices is None or mesh.indices is None:
        return False

    if len(mesh.vertices) == 0 or len(mesh.indices) == 0:
        return False

    return True

def _process_snake3_wheels(csv_path: Path, data: pd.DataFrame) -> None:
    # Extract vertex and index data from CSV
    csv_indices = data[VTX].to_numpy()
    csv_positions = data[[POSITION_X, POSITION_Y, POSITION_Z]].to_numpy()
    csv_texcoords1 = data[[TEXCOORD1_X, TEXCOORD1_Y]].to_numpy()
    csv_texcoords2 = data[[TEXCOORD2_X, TEXCOORD2_Y]].to_numpy()
    csv_texcoords3 = data[[TEXCOORD3_X, TEXCOORD3_Y]].to_numpy()
    csv_texcoords4 = data[[TEXCOORD4_X, TEXCOORD4_Y]].to_numpy()

    # Extract vertex and index count from CSV data
    vertex_count = csv_indices.max() + 1
    index_count = len(csv_indices)

    # Allocate arrays for mesh
    mesh = MeshData()
    mesh.name = 'mesh'
    mesh.vertices = np.empty((vertex_count, 3))
    mesh.normals = np.empty((vertex_count, 3))
    mesh.indices = np.empty(index_count)

    # Loop over data and store unique vertices and fill index array
    offset = 0
    for i in range(index_count):
        idx = int(csv_indices[i])
        if offset <= idx: # Only new vertices
            mesh.vertices[offset] = csv_positions[i, 0:3]
            mesh.normals[offset] = _unpack_normal(csv_texcoords3[i, 0])
            offset += 1
        # Always store the index
        mesh.indices[i] = idx

    _save_usd(csv_path, [mesh])

def _process_snake3_body(csv_path: Path, data: pd.DataFrame) -> None:
    # Extract vertex and index data from CSV
    csv_indices = data[VTX].to_numpy()
    csv_positions = data[[POSITION_X, POSITION_Y, POSITION_Z, POSITION_W]].to_numpy()
    csv_texcoords0 = data[[TEXCOORD0_X, TEXCOORD0_Y, TEXCOORD0_Z, TEXCOORD0_W]].to_numpy()
    csv_texcoords1 = data[[TEXCOORD1_X, TEXCOORD1_Y]].to_numpy()
    csv_texcoords2 = data[[TEXCOORD2_X, TEXCOORD2_Y]].to_numpy()
    csv_texcoords3 = data[[TEXCOORD3_X, TEXCOORD3_Y, TEXCOORD3_Z, TEXCOORD3_W]].to_numpy()
    csv_texcoords4 = data[[TEXCOORD4_X, TEXCOORD4_Y]].to_numpy()

    # Extract vertex and index count from CSV data
    vertex_count = csv_indices.max() + 1
    index_count = len(csv_indices)

    # Allocate arrays for meshes
    mesh = MeshData()
    mesh.name = 'mesh'
    mesh.vertices = np.empty((vertex_count, 3))
    mesh.normals = np.empty((vertex_count, 3))
    mesh.indices = np.empty(index_count)

    mesh_deformed = MeshData()
    mesh_deformed.name = 'deformed'
    mesh_deformed.vertices = np.empty((vertex_count, 3))
    mesh_deformed.normals = np.empty((vertex_count, 3))
    mesh_deformed.indices = np.empty(index_count)

    # Loop over data and store unique vertices and fill index array
    offset = 0
    for i in range(index_count):
        idx = int(csv_indices[i])
        if offset <= idx: # Only new vertices
            mesh.vertices[offset] = csv_positions[i, 0:3]
            mesh.normals[offset] = _unpack_normal(csv_texcoords3[i, 0])
            mesh_deformed.vertices[offset] = _mesh_deformation(csv_positions[i])
            mesh_deformed.normals[offset] = _unpack_normal(csv_texcoords3[i, 0])
            offset += 1
        # Always store the index
        mesh.indices[i] = idx
        mesh_deformed.indices[i] = idx

    _save_usd(csv_path, [mesh, mesh_deformed])

def _unpack_normal(packed: float) -> np.ndarray:
    """
    Unpack float3 from packed float value.
    """
    scaling_factors = np.array([1.0, 0.003906, 0.000015])

    # mul, frc, mad
    unpacked = packed * scaling_factors
    unpacked = np.fmod(unpacked, 1.0)
    unpacked = unpacked * 2.0 - 1.0

    # dp3, rsq, mul
    length_squared = np.dot(unpacked, unpacked)
    inv_length = 1.0 / np.sqrt(length_squared)
    unpacked = unpacked * inv_length

    return unpacked

def _mesh_deformation(position: np.ndarray) -> np.ndarray:
    """
    Apply vertex deformation from packed vertex data.
    """
    scaling_factors = np.array([1.000000, 0.003906, 0.000015])

    # mul, frc, mad
    offset = position[3] * scaling_factors
    offset = offset - np.floor(offset)
    offset = offset * 2.0 - 1.0

    return position[0:3] + offset

if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) != 2:
        print('Usage: python renderdoc_csv_to_usd.py <csv_file>')
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f'Error: File not found: {csv_path}')
        sys.exit(1)

    csv_to_usd(csv_path)
