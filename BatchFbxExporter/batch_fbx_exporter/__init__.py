# -*- coding: utf-8 -*-
"""
Batch FBX Exporter for RenderDoc
Combines batch export functionality with FBX format output
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

__author__ = "Maek"
__date__ = "2025-11-11"

import os
import sys
import time
import struct
import inspect
from textwrap import dedent
from functools import partial
from collections import defaultdict

from PySide2 import QtWidgets, QtCore

import qrenderdoc
import renderdoc as rd

from .batch_dialog import BatchExportDialog
from .progress_dialog import MProgressDialog
from .export_window import ExportLogWindow

FBX_ASCII_TEMPLATE = """
    ; FBX 7.3.0 project file
    ; ----------------------------------------------------

    ; Object definitions
    ;------------------------------------------------------------------

    Definitions:  {

        ObjectType: "Geometry" {
            Count: 1
            PropertyTemplate: "FbxMesh" {
                Properties70:  {
                    P: "Primary Visibility", "bool", "", "",1
                }
            }
        }

        ObjectType: "Model" {
            Count: 1
            PropertyTemplate: "FbxNode" {
                Properties70:  {
                    P: "Visibility", "Visibility", "", "A",1
                }
            }
        }
    }

    ; Object properties
    ;------------------------------------------------------------------

    Objects:  {
        Geometry: 2035541511296, "Geometry::", "Mesh" {
            Vertices: *%(vertices_num)s {
                a: %(vertices)s
            } 
            PolygonVertexIndex: *%(polygons_num)s {
                a: %(polygons)s
            } 
            GeometryVersion: 124
            %(LayerElementNormal)s
            %(LayerElementBiNormal)s
            %(LayerElementTangent)s
            %(LayerElementColor)s
            %(LayerElementUV)s
            %(LayerElementUV2)s
            %(LayerElementUV3)s
            %(LayerElementUV4)s
            %(LayerElementUV5)s
            Layer: 0 {
                Version: 100
                %(LayerElementNormalInsert)s
                %(LayerElementBiNormalInsert)s
                %(LayerElementTangentInsert)s
                %(LayerElementColorInsert)s
                %(LayerElementUVInsert)s
                
            }
            Layer: 1 {
                Version: 100
                %(LayerElementUV2Insert)s
            }
            Layer: 2 {
                Version: 100
                %(LayerElementUV3Insert)s
            }
            Layer: 3 {
                Version: 100
                %(LayerElementUV4Insert)s
            }
            Layer: 4 {
                Version: 100
                %(LayerElementUV5Insert)s
            }
        }
        Model: 2035615390896, "Model::%(model_name)s", "Mesh" {
            Properties70:  {
                P: "DefaultAttributeIndex", "int", "Integer", "",0
            }
        }
    }

    ; Object connections
    ;------------------------------------------------------------------

    Connections:  {
        
        ;Model::pCube1, Model::RootNode
        C: "OO",2035615390896,0
        
        ;Geometry::, Model::pCube1
        C: "OO",2035541511296,2035615390896

    }

    """


# ==================== Data Unpacking Functions ====================

class MeshData(rd.MeshFormat):
    """Extended MeshFormat with additional properties"""
    indexOffset = 0
    name = ''


def unpackData(fmt, data):
    """Unpack vertex data from bytes based on format"""
    formatChars = {}
    #                                 012345678
    formatChars[rd.CompType.UInt]  = "xBHxIxxxL"
    formatChars[rd.CompType.SInt]  = "xbhxixxxl"
    formatChars[rd.CompType.Float] = "xxexfxxxd"  # only 2, 4 and 8 are valid

    # These types have identical decodes, but we might post-process them
    formatChars[rd.CompType.UNorm] = formatChars[rd.CompType.UInt]
    formatChars[rd.CompType.UScaled] = formatChars[rd.CompType.UInt]
    formatChars[rd.CompType.SNorm] = formatChars[rd.CompType.SInt]
    formatChars[rd.CompType.SScaled] = formatChars[rd.CompType.SInt]

    # We need to fetch compCount components
    vertexFormat = str(fmt.compCount) + formatChars[fmt.compType][fmt.compByteWidth]

    # Unpack the data
    value = struct.unpack_from(vertexFormat, data, 0)

    # If the format needs post-processing such as normalisation, do that now
    if fmt.compType == rd.CompType.UNorm:
        divisor = float((2 ** (fmt.compByteWidth * 8)) - 1)
        value = tuple(float(i) / divisor for i in value)
    elif fmt.compType == rd.CompType.SNorm:
        maxNeg = -float(2 ** (fmt.compByteWidth * 8)) / 2
        divisor = float(-(maxNeg-1))
        value = tuple((float(i) if (i == maxNeg) else (float(i) / divisor)) for i in value)

    # If the format is BGRA, swap the two components
    if fmt.BGRAOrder():
        value = tuple(value[i] for i in [2, 1, 0, 3])

    return value


def getMeshInputs(controller, draw):
    """Get vertex and index buffer information for a draw call"""
    state = controller.GetPipelineState()

    # Get the index & vertex buffers, and fixed vertex inputs
    ib = state.GetIBuffer()
    vbs = state.GetVBuffers()
    attrs = state.GetVertexInputs()
    
    meshInputs = []
    
    for attr in attrs:
        # We don't handle instance attributes
        if attr.perInstance:
            continue
        
        meshInput = MeshData()
        meshInput.indexResourceId = ib.resourceId
        meshInput.indexByteOffset = ib.byteOffset
        meshInput.indexByteStride = ib.byteStride
        meshInput.baseVertex = draw.baseVertex
        meshInput.indexOffset = draw.indexOffset
        meshInput.numIndices = draw.numIndices

        # If the draw doesn't use an index buffer, don't use it even if bound
        if not (draw.flags & rd.ActionFlags.Indexed):
            meshInput.indexResourceId = rd.ResourceId.Null()

        # The total offset is the attribute offset from the base of the vertex
        meshInput.vertexByteOffset = attr.byteOffset + vbs[attr.vertexBuffer].byteOffset + draw.vertexOffset * vbs[attr.vertexBuffer].byteStride
        meshInput.format = attr.format
        meshInput.vertexResourceId = vbs[attr.vertexBuffer].resourceId
        meshInput.vertexByteStride = vbs[attr.vertexBuffer].byteStride
        meshInput.name = attr.name

        meshInputs.append(meshInput)

    return meshInputs


def getIndices(controller, mesh):
    """Extract indices from index buffer"""
    # Get the character for the width of index
    indexFormat = 'B'
    if mesh.indexByteStride == 2:
        indexFormat = 'H'
    elif mesh.indexByteStride == 4:
        indexFormat = 'I'

    # Duplicate the format by the number of indices
    indexFormat = str(mesh.numIndices) + indexFormat

    # If we have an index buffer
    if mesh.indexResourceId != rd.ResourceId.Null():
        # Fetch the data
        ibdata = controller.GetBufferData(mesh.indexResourceId, mesh.indexByteOffset, 0)
        # Unpack all the indices, starting from the first index to fetch
        offset = mesh.indexOffset * mesh.indexByteStride
        indices = struct.unpack_from(indexFormat, ibdata, offset)

        # Apply the baseVertex offset
        return [i + mesh.baseVertex for i in indices]
    else:
        # With no index buffer, just generate a range
        return tuple(range(mesh.numIndices))


def saveTexture(resourceId, eventId, output_folder, controller):
    """Save a single texture to PNG"""
    texsave = rd.TextureSave()
    texsave.resourceId = resourceId
    if texsave.resourceId == rd.ResourceId.Null():
        return False

    filename = str(int(texsave.resourceId))
    texsave.mip = 0
    texsave.slice.sliceIndex = 0
    texsave.alpha = rd.AlphaMapping.Preserve
    texsave.destType = rd.FileType.PNG
    
    event_folder = os.path.join(output_folder, str(eventId))
    if not os.path.exists(event_folder):
        os.makedirs(event_folder)

    outTexPath = os.path.join(event_folder, filename + ".png")
    controller.SaveTexture(texsave, outTexPath)
    print("Saved texture: {0}".format(outTexPath))
    return True


# ==================== FBX Export Functions ====================

def export_fbx(save_path, mapper, data, attr_list, controller):
    """Export mesh data to FBX format"""
    if not data:
        return False

    # Get filename without extension (e.g., "1" from "1.fbx")
    base_name = os.path.basename(os.path.splitext(save_path)[0])
    # Format as mesh_N for FBX internal naming
    save_name = "mesh_{0}".format(base_name)

    # We'll decode the first three indices making up a triangle
    idx_dict = data["IDX"]
    value_dict = defaultdict(list)
    vertex_data = defaultdict(dict)

    for i, idx in enumerate(idx_dict):
        for attr in attr_list:
            if attr in data:
                value = data[attr][i]
                value_dict[attr].append(value)
                if idx not in vertex_data[attr]:
                    vertex_data[attr][idx] = value

    ARGS = {
        "model_name": save_name,
        "LayerElementNormal": "",
        "LayerElementNormalInsert": "",
        "LayerElementBiNormal": "",
        "LayerElementBiNormalInsert": "",
        "LayerElementTangent": "",
        "LayerElementTangentInsert": "",
        "LayerElementColor": "",
        "LayerElementColorInsert": "",
        "LayerElementUV": "",
        "LayerElementUVInsert": "",
        "LayerElementUV2": "",
        "LayerElementUV2Insert": "",
        "LayerElementUV3": "",
        "LayerElementUV3Insert": "",
        "LayerElementUV4": "",
        "LayerElementUV4Insert": "",
        "LayerElementUV5": "",
        "LayerElementUV5Insert": "",
    }

    POSITION = mapper.get("POSITION", "")
    NORMAL = mapper.get("NORMAL", "")
    BINORMAL = mapper.get("BINORMAL", "")
    TANGENT = mapper.get("TANGENT", "")
    COLOR = mapper.get("COLOR", "")
    UV = mapper.get("UV", "")
    UV2 = mapper.get("UV2", "")
    UV3 = mapper.get("UV3", "")
    UV4 = mapper.get("UV4", "")
    UV5 = mapper.get("UV5", "")

    if not vertex_data.get(POSITION):
        return False

    min_poly = min(idx_dict)
    idx_list = [idx - min_poly for idx in idx_dict]
    idx_len = len(idx_list)

    class ProcessHandler(object):
        def run(self):
            for name, func in inspect.getmembers(self, inspect.isroutine):
                if name.startswith("run_"):
                    func()

        def run_vertices(self):
            vertices = [str(v) for idx, values in sorted(vertex_data[POSITION].items()) for v in values[:3]]
            ARGS["vertices"] = ",".join(vertices)
            ARGS["vertices_num"] = len(vertices)

        def run_polygons(self):
            polygons = [str(idx ^ -1 if i % 3 == 2 else idx) for i, idx in enumerate(idx_list)]
            ARGS["polygons"] = ",".join(polygons)
            ARGS["polygons_num"] = len(polygons)

        def run_normals(self):
            if not NORMAL or not vertex_data.get(NORMAL):
                return

            normals = [str(v) for values in value_dict[NORMAL] for v in values[:3]]

            ARGS["LayerElementNormal"] = """
                LayerElementNormal: 0 {
                    Version: 101
                    Name: ""
                    MappingInformationType: "ByPolygonVertex"
                    ReferenceInformationType: "Direct"
                    Normals: *%(normals_num)s {
                        a: %(normals)s
                    } 
                }
            """ % {
                "normals": ",".join(normals),
                "normals_num": len(normals),
            }
            ARGS["LayerElementNormalInsert"] = """
                LayerElement:  {
                        Type: "LayerElementNormal"
                    TypedIndex: 0
                }
            """

        def run_binormals(self):
            if not BINORMAL or not vertex_data.get(BINORMAL):
                return
            binormals = [str(-v) for values in value_dict[BINORMAL] for v in values[:3]]

            ARGS["LayerElementBiNormal"] = """
                LayerElementBinormal: 0 {
                    Version: 101
                    Name: "map1"
                    MappingInformationType: "ByVertice"
                    ReferenceInformationType: "Direct"
                    Binormals: *%(binormals_num)s {
                        a: %(binormals)s
                    } 
                    BinormalsW: *%(binormalsW_num)s {
                        a: %(binormalsW)s
                    } 
                }
            """ % {
                "binormals": ",".join(binormals),
                "binormals_num": len(binormals),
                "binormalsW": ",".join(["1" for i in range(idx_len)]),
                "binormalsW_num": idx_len,
            }
            ARGS["LayerElementBiNormalInsert"] = """
                LayerElement:  {
                        Type: "LayerElementBinormal"
                    TypedIndex: 0
                }
            """

        def run_tangents(self):
            if not TANGENT or not vertex_data.get(TANGENT):
                return

            tangents = [str(v) for values in value_dict[TANGENT] for v in values[:3]]

            ARGS["LayerElementTangent"] = """
                LayerElementTangent: 0 {
                    Version: 101
                    Name: "map1"
                    MappingInformationType: "ByPolygonVertex"
                    ReferenceInformationType: "Direct"
                    Tangents: *%(tangents_num)s {
                        a: %(tangents)s
                    } 
                }
            """ % {
                "tangents": ",".join(tangents),
                "tangents_num": len(tangents),
            }

            ARGS["LayerElementTangentInsert"] = """
                    LayerElement:  {
                        Type: "LayerElementTangent"
                        TypedIndex: 0
                    }
            """

        def run_color(self):
            if not COLOR or not vertex_data.get(COLOR):
                return

            colors = [
                str(v)
                for values in value_dict[COLOR]
                for i, v in enumerate(values, 1)
            ]

            ARGS["LayerElementColor"] = """
                LayerElementColor: 0 {
                    Version: 101
                    Name: "colorSet1"
                    MappingInformationType: "ByPolygonVertex"
                    ReferenceInformationType: "IndexToDirect"
                    Colors: *%(colors_num)s {
                        a: %(colors)s
                    } 
                    ColorIndex: *%(colors_indices_num)s {
                        a: %(colors_indices)s
                    } 
                }
            """ % {
                "colors": ",".join(colors),
                "colors_num": len(colors),
                "colors_indices": ",".join([str(i) for i in range(idx_len)]),
                "colors_indices_num": idx_len,
            }
            ARGS["LayerElementColorInsert"] = """
                LayerElement:  {
                    Type: "LayerElementColor"
                    TypedIndex: 0
                }
            """

        def run_uv(self):
            if not UV or not vertex_data.get(UV):
                return

            uvs_indices = ",".join([str(idx) for idx in idx_list])
            uvs = [
                # NOTE flip y axis
                str(1 - v if i else v)
                for idx, values in sorted(vertex_data[UV].items())
                for i, v in enumerate(values[:2])
            ]

            ARGS["LayerElementUV"] = """
                LayerElementUV: 0 {
                    Version: 101
                    Name: "map1"
                    MappingInformationType: "ByPolygonVertex"
                    ReferenceInformationType: "IndexToDirect"
                    UV: *%(uvs_num)s {
                        a: %(uvs)s
                    } 
                    UVIndex: *%(uvs_indices_num)s {
                        a: %(uvs_indices)s
                    } 
                }
            """ % {
                "uvs": ",".join(uvs),
                "uvs_num": len(uvs),
                "uvs_indices": uvs_indices,
                "uvs_indices_num": idx_len,
            }

            ARGS["LayerElementUVInsert"] = """
                LayerElement:  {
                    Type: "LayerElementUV"
                    TypedIndex: 0
                }
            """

        def run_uv2(self):
            if not UV2 or not vertex_data.get(UV2):
                return

            uvs_indices = ",".join([str(idx) for idx in idx_list])
            uvs = [
                # NOTE flip y axis
                str(1 - v if i else v)
                for idx, values in sorted(vertex_data[UV2].items())
                for i, v in enumerate(values[:2])
            ]

            ARGS["LayerElementUV2"] = """
                LayerElementUV: 1 {
                    Version: 101
                    Name: "map2"
                    MappingInformationType: "ByPolygonVertex"
                    ReferenceInformationType: "IndexToDirect"
                    UV: *%(uvs_num)s {
                        a: %(uvs)s
                    } 
                    UVIndex: *%(uvs_indices_num)s {
                        a: %(uvs_indices)s
                    } 
                }
            """ % {
                "uvs": ",".join(uvs),
                "uvs_num": len(uvs),
                "uvs_indices": uvs_indices,
                "uvs_indices_num": idx_len,
            }

            ARGS["LayerElementUV2Insert"] = """
                LayerElement:  {
                    Type: "LayerElementUV"
                    TypedIndex: 1
                }
            """

        def run_uv3(self):
            if not UV3 or not vertex_data.get(UV3):
                return

            uvs_indices = ",".join([str(idx) for idx in idx_list])
            uvs = [
                # NOTE flip y axis
                str(1 - v if i else v)
                for idx, values in sorted(vertex_data[UV3].items())
                for i, v in enumerate(values[:2])
            ]

            ARGS["LayerElementUV3"] = """
                LayerElementUV: 2 {
                    Version: 101
                    Name: "map3"
                    MappingInformationType: "ByPolygonVertex"
                    ReferenceInformationType: "IndexToDirect"
                    UV: *%(uvs_num)s {
                        a: %(uvs)s
                    } 
                    UVIndex: *%(uvs_indices_num)s {
                        a: %(uvs_indices)s
                    } 
                }
            """ % {
                "uvs": ",".join(uvs),
                "uvs_num": len(uvs),
                "uvs_indices": uvs_indices,
                "uvs_indices_num": idx_len,
            }

            ARGS["LayerElementUV3Insert"] = """
                LayerElement:  {
                    Type: "LayerElementUV"
                    TypedIndex: 2
                }
            """

        def run_uv4(self):
            if not UV4 or not vertex_data.get(UV4):
                return

            uvs_indices = ",".join([str(idx) for idx in idx_list])
            uvs = [
                # NOTE flip y axis
                str(1 - v if i else v)
                for idx, values in sorted(vertex_data[UV4].items())
                for i, v in enumerate(values[:2])
            ]

            ARGS["LayerElementUV4"] = """
                LayerElementUV: 3 {
                    Version: 101
                    Name: "map4"
                    MappingInformationType: "ByPolygonVertex"
                    ReferenceInformationType: "IndexToDirect"
                    UV: *%(uvs_num)s {
                        a: %(uvs)s
                    } 
                    UVIndex: *%(uvs_indices_num)s {
                        a: %(uvs_indices)s
                    } 
                }
            """ % {
                "uvs": ",".join(uvs),
                "uvs_num": len(uvs),
                "uvs_indices": uvs_indices,
                "uvs_indices_num": idx_len,
            }

            ARGS["LayerElementUV4Insert"] = """
                LayerElement:  {
                    Type: "LayerElementUV"
                    TypedIndex: 3
                }
            """

        def run_uv5(self):
            if not UV5 or not vertex_data.get(UV5):
                return

            uvs_indices = ",".join([str(idx) for idx in idx_list])
            uvs = [
                # NOTE flip y axis
                str(1 - v if i else v)
                for idx, values in sorted(vertex_data[UV5].items())
                for i, v in enumerate(values[:2])
            ]

            ARGS["LayerElementUV5"] = """
                LayerElementUV: 4 {
                    Version: 101
                    Name: "map5"
                    MappingInformationType: "ByPolygonVertex"
                    ReferenceInformationType: "IndexToDirect"
                    UV: *%(uvs_num)s {
                        a: %(uvs)s
                    } 
                    UVIndex: *%(uvs_indices_num)s {
                        a: %(uvs_indices)s
                    } 
                }
            """ % {
                "uvs": ",".join(uvs),
                "uvs_num": len(uvs),
                "uvs_indices": uvs_indices,
                "uvs_indices_num": idx_len,
            }

            ARGS["LayerElementUV5Insert"] = """
                LayerElement:  {
                    Type: "LayerElementUV"
                    TypedIndex: 4
                }
            """

    handler = ProcessHandler()
    handler.run()

    fbx = FBX_ASCII_TEMPLATE % ARGS

    with open(save_path, "w") as f:
        f.write(dedent(fbx).strip())
    
    return True


# ==================== Matrix Transform Functions ====================

def read_matrix_from_renderdoc(controller, target_set, target_binding, variable_name, log_func=None):
    """Read transformation matrix from RenderDoc constant buffer
    
    Args:
        controller: RenderDoc controller
        target_set: Descriptor set number
        target_binding: Binding number
        variable_name: Variable name in constant buffer
        log_func: Optional logging function
    
    Returns:
        tuple of 16 floats (4x4 matrix) or None if failed
    """
    def log(msg):
        if log_func:
            log_func(msg)
        else:
            print(msg)
    
    try:
        import struct
        
        # Get pipeline state
        state = controller.GetPipelineState()
        if not state:
            log("Error: No pipeline state")
            return None
        
        # Get shader reflection
        shader_refl = state.GetShaderReflection(rd.ShaderStage.Vertex)
        if not shader_refl:
            log("Error: No shader reflection")
            return None
        
        # Get runtime constant blocks
        runtime_cbs = state.GetConstantBlocks(rd.ShaderStage.Vertex)
        
        # Search through constant blocks in shader reflection
        for cb_index, cb_refl in enumerate(shader_refl.constantBlocks):
            # Check if this CB matches our target set and binding
            if not hasattr(cb_refl, 'fixedBindSetOrSpace') or not hasattr(cb_refl, 'fixedBindNumber'):
                continue
            
            cb_set = cb_refl.fixedBindSetOrSpace
            cb_binding = cb_refl.fixedBindNumber
            
            if cb_set == target_set and cb_binding == target_binding:
                # Search for the variable
                var_found = False
                var_offset = 0
                
                for var in cb_refl.variables:
                    if hasattr(var, 'name') and var.name == variable_name:
                        var_found = True
                        if hasattr(var, 'offset'):
                            var_offset = var.offset
                        break
                
                if not var_found:
                    log("Variable '{0}' not found in buffer {1}".format(variable_name, cb_refl.name))
                    return None
                
                # Read buffer data from runtime constant block
                if cb_index >= len(runtime_cbs):
                    log("Error: Runtime CB index out of range")
                    return None
                
                runtime_cb = runtime_cbs[cb_index]
                if not hasattr(runtime_cb, 'descriptor'):
                    log("Error: Runtime CB has no descriptor")
                    return None
                
                desc = runtime_cb.descriptor
                if not hasattr(desc, 'resource'):
                    log("Error: Descriptor has no resource")
                    return None
                
                res_id = desc.resource
                buf_offset = desc.byteOffset if hasattr(desc, 'byteOffset') else 0
                buf_size = desc.byteSize if hasattr(desc, 'byteSize') else cb_refl.byteSize
                
                # Read buffer data
                buffer_data = controller.GetBufferData(res_id, buf_offset, buf_size)
                
                if len(buffer_data) < var_offset + 64:
                    log("Error: Not enough data in buffer")
                    return None
                
                # Extract matrix at variable offset
                matrix = struct.unpack('16f', buffer_data[var_offset:var_offset+64])
                
                # Transpose matrix (Vulkan uses column-major, we need row-major)
                # Original: column-major [col0, col1, col2, col3]
                # Need: row-major [row0, row1, row2, row3]
                transposed = (
                    matrix[0], matrix[4], matrix[8], matrix[12],   # row 0
                    matrix[1], matrix[5], matrix[9], matrix[13],   # row 1
                    matrix[2], matrix[6], matrix[10], matrix[14],  # row 2
                    matrix[3], matrix[7], matrix[11], matrix[15]   # row 3
                )
                return transposed
        
        log("No constant buffer found with Set={0}, Binding={1}".format(target_set, target_binding))
        return None
        
    except Exception as e:
        log("Error reading matrix: {0}".format(str(e)))
        return None


def transform_vertices_with_matrix(data, matrix, mapper):
    """Apply 4x4 transform matrix to vertex data
    
    Args:
        data: Dictionary with vertex data, keys are actual attribute names (e.g., "_input0")
        matrix: 4x4 transformation matrix (16 floats)
        mapper: Dictionary mapping semantic names to actual attribute names
                e.g., {'POSITION': '_input0', 'NORMAL': '_input1'}
    """
    if not matrix:
        return data
    
    # Get actual attribute names from mapper
    position_key = mapper.get('POSITION', '')
    normal_key = mapper.get('NORMAL', '')
    tangent_key = mapper.get('TANGENT', '')
    binormal_key = mapper.get('BINORMAL', '')
    
    # Transform POSITION
    if position_key and position_key in data:
        transformed_positions = []
        for pos in data[position_key]:
            x, y, z = pos[:3] if len(pos) >= 3 else (pos[0] if len(pos) >= 1 else 0, pos[1] if len(pos) >= 2 else 0, 0)
            
            # Matrix multiplication: pos_world = Matrix × pos_local
            # Matrix is in row-major order: [m00,m01,m02,m03, m10,m11,m12,m13, ...]
            x_new = matrix[0]*x + matrix[1]*y + matrix[2]*z  + matrix[3]
            y_new = matrix[4]*x + matrix[5]*y + matrix[6]*z  + matrix[7]
            z_new = matrix[8]*x + matrix[9]*y + matrix[10]*z + matrix[11]
            
            transformed_positions.append((x_new, y_new, z_new))
        
        data[position_key] = transformed_positions
    
    # Transform NORMAL (rotation only, no translation)
    if normal_key and normal_key in data:
        transformed_normals = []
        for norm in data[normal_key]:
            x, y, z = norm[:3] if len(norm) >= 3 else (norm[0] if len(norm) >= 1 else 0, norm[1] if len(norm) >= 2 else 0, 0)
            
            # Use rotation part only (3x3 upper-left)
            x_new = matrix[0]*x + matrix[1]*y + matrix[2]*z
            y_new = matrix[4]*x + matrix[5]*y + matrix[6]*z
            z_new = matrix[8]*x + matrix[9]*y + matrix[10]*z
            
            # Normalize
            length = (x_new*x_new + y_new*y_new + z_new*z_new) ** 0.5
            if length > 0.0001:
                transformed_normals.append((x_new/length, y_new/length, z_new/length))
            else:
                transformed_normals.append(norm)
        
        data[normal_key] = transformed_normals
    
    # Transform TANGENT (rotation only)
    if tangent_key and tangent_key in data:
        transformed_tangents = []
        for tang in data[tangent_key]:
            x, y, z = tang[:3] if len(tang) >= 3 else (tang[0] if len(tang) >= 1 else 0, tang[1] if len(tang) >= 2 else 0, 0)
            
            x_new = matrix[0]*x + matrix[1]*y + matrix[2]*z
            y_new = matrix[4]*x + matrix[5]*y + matrix[6]*z
            z_new = matrix[8]*x + matrix[9]*y + matrix[10]*z
            
            length = (x_new*x_new + y_new*y_new + z_new*z_new) ** 0.5
            if length > 0.0001:
                transformed_tangents.append((x_new/length, y_new/length, z_new/length))
            else:
                transformed_tangents.append(tang)
        
        data[tangent_key] = transformed_tangents
    
    # Transform BINORMAL (rotation only)
    if binormal_key and binormal_key in data:
        transformed_binormals = []
        for binorm in data[binormal_key]:
            x, y, z = binorm[:3] if len(binorm) >= 3 else (binorm[0] if len(binorm) >= 1 else 0, binorm[1] if len(binorm) >= 2 else 0, 0)
            
            x_new = matrix[0]*x + matrix[1]*y + matrix[2]*z
            y_new = matrix[4]*x + matrix[5]*y + matrix[6]*z
            z_new = matrix[8]*x + matrix[9]*y + matrix[10]*z
            
            length = (x_new*x_new + y_new*y_new + z_new*z_new) ** 0.5
            if length > 0.0001:
                transformed_binormals.append((x_new/length, y_new/length, z_new/length))
            else:
                transformed_binormals.append(binorm)
        
        data[binormal_key] = transformed_binormals
    
    return data


# ==================== Batch Export Logic ====================

def export_draw_call(controller, draw, mapper, output_folder, log_window=None, draw_index=None, matrix_config=None):
    """Export a single draw call to FBX and textures
    
    Args:
        controller: RenderDoc controller
        draw: Draw call to export
        mapper: Attribute mapping dictionary
        output_folder: Output directory
        log_window: Optional log window for output
        draw_index: DEPRECATED - no longer used, eventId is used instead
        matrix_config: Optional dict with 'set', 'binding', 'variable' for real-time matrix reading
    """
    def log(msg):
        if log_window:
            log_window.log(msg)
        print(msg)
    
    try:
        log("[EventID {0}] Processing...".format(draw.eventId))
        
        # Move to that draw
        controller.SetFrameEvent(draw.eventId, True)
        
        # Get mesh inputs
        meshInputs = getMeshInputs(controller, draw)
        
        if not meshInputs:
            log("[EventID {0}] ✗ No mesh inputs found, skipping".format(draw.eventId))
            return False
        
        # Extract mesh data
        indices = getIndices(controller, meshInputs[0])
        log("[EventID {0}] Found {1} indices".format(draw.eventId, len(indices)))
        
        data = defaultdict(list)
        attr_list = set()
        
        # Store indices
        data["IDX"] = indices
        
        # ✅ 性能优化：批量读取顶点缓冲区（而不是逐个顶点读取）
        # 旧方法：每个顶点调用一次GetBufferData，1536个顶点×5个属性 = 7680次API调用
        # 新方法：每个属性只调用一次GetBufferData，5次API调用
        # 性能提升：约1000-1500倍！
        
        log("[EventID {0}] Reading vertex buffers...".format(draw.eventId))
        
        for attr in meshInputs:
            if not attr.format.Special():
                attr_list.add(attr.name)
                
                # 计算需要读取的缓冲区大小
                max_idx = max(indices) if indices else 0
                buffer_size = (max_idx + 1) * attr.vertexByteStride
                
                # ✅ 一次性读取整个缓冲区（关键优化）
                try:
                    full_buffer = controller.GetBufferData(
                        attr.vertexResourceId, 
                        attr.vertexByteOffset, 
                        buffer_size
                    )
                    
                    # 在内存中解析每个顶点（非常快）
                    for idx in indices:
                        offset_in_buffer = idx * attr.vertexByteStride
                        value = unpackData(attr.format, full_buffer[offset_in_buffer:])
                        data[attr.name].append(value)
                        
                except Exception as e:
                    log("[EventID {0}] Warning: Failed to read {1}: {2}".format(
                        draw.eventId, attr.name, str(e)))
                    # 如果批量读取失败，回退到逐个读取
                    for idx in indices:
                        try:
                            offset = attr.vertexByteOffset + attr.vertexByteStride * idx
                            buffer_data = controller.GetBufferData(attr.vertexResourceId, offset, 0)
                            value = unpackData(attr.format, buffer_data)
                            data[attr.name].append(value)
                        except:
                            pass
        
        log("[EventID {0}] Extracted attributes: {1}".format(draw.eventId, ", ".join(sorted(attr_list))))
        
        # Read and apply transform matrix in real-time if config provided
        if matrix_config:
            log("[EventID {0}] Reading transform matrix from RenderDoc...".format(draw.eventId))
            log("[EventID {0}]   Set: {1}, Binding: {2}, Variable: {3}".format(
                draw.eventId, matrix_config['set'], matrix_config['binding'], matrix_config['variable']))
            
            # Read matrix for this specific EventID
            transform_matrix = read_matrix_from_renderdoc(
                controller,
                matrix_config['set'],
                matrix_config['binding'],
                matrix_config['variable'],
                log
            )
            
            if transform_matrix:
                # 显示矩阵值
                log("[EventID {0}] Matrix: [{1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}]".format(
                    draw.eventId, transform_matrix[0], transform_matrix[1], transform_matrix[2], transform_matrix[3]))
                log("[EventID {0}]         [{1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}]".format(
                    draw.eventId, transform_matrix[4], transform_matrix[5], transform_matrix[6], transform_matrix[7]))
                log("[EventID {0}]         [{1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}]".format(
                    draw.eventId, transform_matrix[8], transform_matrix[9], transform_matrix[10], transform_matrix[11]))
                log("[EventID {0}]         [{1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}]".format(
                    draw.eventId, transform_matrix[12], transform_matrix[13], transform_matrix[14], transform_matrix[15]))
                
                # Get the actual position attribute name from mapper
                position_key = mapper.get('POSITION', '')
                log("[EventID {0}] Position attribute mapped to: '{1}'".format(draw.eventId, position_key))
                
                # 显示变换前的第一个顶点
                if position_key and position_key in data and len(data[position_key]) > 0:
                    first_pos = data[position_key][0]
                    log("[EventID {0}] First vertex before: ({1:.3f}, {2:.3f}, {3:.3f})".format(
                        draw.eventId, first_pos[0], first_pos[1], first_pos[2]))
                
                # Apply transformation
                data = transform_vertices_with_matrix(data, transform_matrix, mapper)
                
                # 显示变换后的第一个顶点
                if position_key and position_key in data and len(data[position_key]) > 0:
                    first_pos = data[position_key][0]
                    log("[EventID {0}] First vertex after:  ({1:.3f}, {2:.3f}, {3:.3f})".format(
                        draw.eventId, first_pos[0], first_pos[1], first_pos[2]))
                
                log("[EventID {0}] ✓ Transform applied".format(draw.eventId))
            else:
                log("[EventID {0}] ⚠ Failed to read matrix, using original coordinates".format(draw.eventId))
        else:
            log("[EventID {0}] ℹ No transform config provided (using original coordinates)".format(draw.eventId))
        
        # Export textures
        state = controller.GetPipelineState()
        texture_count = 0
        
        # Export fragment shader textures
        try:
            usedDescriptors = state.GetReadOnlyResources(rd.ShaderStage.Fragment)
            for usedDescriptor in usedDescriptors:
                res = usedDescriptor.descriptor.resource
                if saveTexture(res, draw.eventId, output_folder, controller):
                    texture_count += 1
        except:
            pass
        
        # Export render target textures
        try:
            for output in draw.outputs:
                if saveTexture(output, draw.eventId, output_folder, controller):
                    texture_count += 1
        except:
            pass
        
        if texture_count > 0:
            log("[EventID {0}] Exported {1} texture(s)".format(draw.eventId, texture_count))
        
        # Export FBX
        event_folder = os.path.join(output_folder, str(draw.eventId))
        if not os.path.exists(event_folder):
            os.makedirs(event_folder)
        
        # Use eventId for filename (e.g., 832.fbx)
        # This ensures FBX internal mesh name will be mesh_832
        fbx_filename = "{0}.fbx".format(draw.eventId)
        fbx_path = os.path.join(event_folder, fbx_filename)
        success = export_fbx(fbx_path, mapper, data, attr_list, controller)
        
        if success:
            log("[EventID {0}] ✓ Exported to: {1}".format(draw.eventId, fbx_path))
        else:
            log("[EventID {0}] ✗ Failed to export FBX".format(draw.eventId))
        
        return success
        
    except Exception as e:
        log("[EventID {0}] ✗ Error: {1}".format(draw.eventId, str(e)))
        import traceback
        log(traceback.format_exc())
        return False


def find_draws_in_range(controller, start_index, end_index):
    """Find all draw calls within the specified index range"""
    draws_in_range = []
    
    def recursive_search(draw):
        if start_index <= draw.eventId <= end_index:
            # Check if it's an actual draw call (has vertices)
            if draw.numIndices > 0 or draw.numInstances > 0:
                draws_in_range.append(draw)
        
        for child in draw.children:
            recursive_search(child)
    
    for root_draw in controller.GetRootActions():
        recursive_search(root_draw)
    
    return draws_in_range


# ==================== Extension Registration ====================

def error_log(func):
    """Decorator for error logging"""
    def wrapper(pyrenderdoc, data):
        manager = pyrenderdoc.Extensions()
        try:
            func(pyrenderdoc, data)
        except:
            import traceback
            manager.MessageDialog("Batch FBX Export Failed\n%s" % traceback.format_exc(), "Error")
    return wrapper


@error_log
def prepare_batch_export(pyrenderdoc, data):
    """Prepare and show the batch export dialog"""
    manager = pyrenderdoc.Extensions()
    
    mqt = manager.GetMiniQtHelper()
    dialog = BatchExportDialog(mqt, pyrenderdoc)
    
    # Show configuration dialog
    if not mqt.ShowWidgetAsDialog(dialog.init_ui()):
        return
    
    # Create log window
    log_window = ExportLogWindow(
        dialog.start_index,
        dialog.end_index,
        dialog.output_folder
    )
    
    # Create worker thread
    from .export_worker import ExportWorker
    
    # Get matrix config (may be None if widget failed to load or disabled)
    matrix_config = getattr(dialog, 'matrix_config', None)
    
    worker = ExportWorker(
        pyrenderdoc,
        dialog.start_index,
        dialog.end_index,
        dialog.output_folder,
        dialog.mapper,
        find_draws_in_range,   # Pass function reference
        export_draw_call,      # Pass function reference
        matrix_config           # Pass matrix config for real-time reading
    )
    
    # Connect worker signals to log window
    worker.log_signal.connect(log_window.append_log)
    worker.progress_signal.connect(log_window.update_progress)
    worker.finished_signal.connect(log_window.on_export_finished)
    
    # Connect cancel button to worker with confirmation
    def cancel_export():
        reply = QtWidgets.QMessageBox.question(
            log_window, 
            'Cancel Export',
            'Are you sure you want to cancel the export?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            worker.cancel()
            log_window.on_cancel()
    
    log_window.cancel_button.clicked.disconnect()
    log_window.cancel_button.clicked.connect(cancel_export)
    
    # Show window (non-modal)
    log_window.show()
    log_window.raise_()
    log_window.activateWindow()
    log_window.setWindowState(log_window.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
    
    log_window.append_log("Initializing export process...")
    log_window.append_log("Starting background worker thread...")
    
    # Start worker thread (non-blocking!)
    worker.start()
    
    log_window.append_log("✓ Worker thread started successfully")
    log_window.append_log("UI will remain responsive during export\n")


def register(version, pyrenderdoc):
    """Register the extension with RenderDoc"""
    print("Registering Batch FBX Exporter extension for RenderDoc {}".format(version))
    pyrenderdoc.Extensions().RegisterWindowMenu(
        qrenderdoc.WindowMenu.Window, 
        ["Batch FBX Exporter"], 
        prepare_batch_export
    )


def unregister():
    """Unregister the extension"""
    print("Unregistering Batch FBX Exporter extension")

