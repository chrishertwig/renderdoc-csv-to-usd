import csv
import math
import os
import sys
from pathlib import Path
from typing import Tuple

from pxr import Gf, Sdf, Usd, UsdGeom, Vt


def csv_to_usd(csv_path: Path) -> None:
    vertices = []
    texcoords0 = []
    texcoords1 = []
    normals = []
    indices = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            vtx = int(row['VTX'])
            idx = int(row[' IDX'])
            position_x = float(row[' POSITION.x'])
            position_y = float(row[' POSITION.y'])
            position_z = float(row[' POSITION.z'])
            texcoord1_x = float(row[' TEXCOORD1.x'])
            texcoord1_y = float(row[' TEXCOORD1.y'])
            texcoord2_x = float(row[' TEXCOORD2.x'])
            texcoord2_y = float(row[' TEXCOORD2.y'])
            texcoord3_x = float(row[' TEXCOORD3.x'])
            texcoord3_y = float(row[' TEXCOORD3.y'])
            texcoord4_x = float(row[' TEXCOORD4.x'])
            texcoord4_y = float(row[' TEXCOORD4.y'])

            # Store vertex in vertices list if we haven't encountered it before
            if len(vertices) <= idx:
                vertices.append((position_x, position_y, position_z))
                texcoords0.append((texcoord1_x, texcoord1_y))
                texcoords1.append((texcoord2_x, texcoord2_y))
                normals.append(unpack_normal(texcoord3_x))

            indices.append(idx)

    usd_path = csv_path.with_suffix('.usd')
    stage = Usd.Stage.CreateNew(usd_path.as_posix())
    UsdGeom.SetStageMetersPerUnit(stage, 1)

    mesh_path = f'/{csv_path.stem}'
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)

    mesh.CreatePointsAttr(Vt.Vec3fArray(vertices))
    mesh.CreateNormalsAttr(Vt.Vec3fArray(normals))
    mesh.CreateFaceVertexCountsAttr([3] * (len(indices) // 3))
    mesh.CreateFaceVertexIndicesAttr(indices)

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.Save()
    print(f"USD file created: {usd_path}")


def unpack_normal(packed: float) -> Tuple[float, ...]:
    """
    Unpack float3 from packaed float value.

    mul r0.xyz, v3.xxxx, l(0.000015, 1.000000, 0.003906, 0.000000)
    frc r0.xyz, r0.xyzx
    mad r0.xyz, r0.xyzx, l(2.000000, 2.000000, 2.000000, 0.000000), l(-1.000000, -1.000000, -1.000000, 0.000000)
    dp3 r0.w, r0.xyzx, r0.xyzx
    rsq r0.w, r0.w
    mul r0.xyz, r0.wwww, r0.xyzx
    """
    # mul
    scaling_factors = (0.000015, 1.000000, 0.003906)
    r0 = tuple(packed * factor for factor in scaling_factors)

    # frc
    r0 = tuple(x - math.floor(x) for x in r0)

    # mad
    r0 = tuple(x * 2.0 - 1.0 for x in r0)

    # dp3, rsq, mul
    length_squared = sum(x * x for x in r0)
    if length_squared > 0:
        inv_length = 1.0 / math.sqrt(length_squared)
        r0 = tuple(x * inv_length for x in r0)

    return r0

def mesh_deformation(position: Tuple[float, ...], packed: Tuple[float, ...]) -> Tuple[float, ...]:
    offsets = tuple(x * 127.996094 % 1 for x in packed)
    deformed = (position[0] + offsets[0], position[1] + offsets[1], position[2] + offsets[2])
    return deformed

if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) != 2:
        print("Usage: python renderdoc_csv_to_usd.py <csv_file> [usd_file]")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    csv_to_usd(csv_path)
