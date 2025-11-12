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

    save_name = os.path.basename(os.path.splitext(save_path)[0])

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


# ==================== Batch Export Logic ====================

def export_draw_call(controller, draw, mapper, output_folder, log_window=None):
    """Export a single draw call to FBX and textures"""
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
        
        fbx_path = os.path.join(event_folder, "model.fbx")
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
    dialog = BatchExportDialog(mqt)
    
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
    worker = ExportWorker(
        pyrenderdoc,
        dialog.start_index,
        dialog.end_index,
        dialog.output_folder,
        dialog.mapper,
        find_draws_in_range,  # Pass function reference
        export_draw_call       # Pass function reference
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

