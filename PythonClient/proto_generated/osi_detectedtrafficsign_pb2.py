# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: osi_detectedtrafficsign.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import osi_common_pb2
import osi_trafficsign_pb2
import osi_detectedobject_pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='osi_detectedtrafficsign.proto',
  package='osi3',
  serialized_pb=_b('\n\x1dosi_detectedtrafficsign.proto\x12\x04osi3\x1a\x10osi_common.proto\x1a\x15osi_trafficsign.proto\x1a\x18osi_detectedobject.proto\"\xb9\t\n\x13\x44\x65tectedTrafficSign\x12(\n\x06header\x18\x01 \x01(\x0b\x32\x18.osi3.DetectedItemHeader\x12=\n\tmain_sign\x18\x02 \x01(\x0b\x32*.osi3.DetectedTrafficSign.DetectedMainSign\x12O\n\x12supplementary_sign\x18\x03 \x03(\x0b\x32\x33.osi3.DetectedTrafficSign.DetectedSupplementarySign\x1a\x9a\x05\n\x10\x44\x65tectedMainSign\x12O\n\tcandidate\x18\x01 \x03(\x0b\x32<.osi3.DetectedTrafficSign.DetectedMainSign.CandidateMainSign\x12\"\n\x04\x62\x61se\x18\x02 \x01(\x0b\x32\x14.osi3.BaseStationary\x12\'\n\tbase_rmse\x18\x03 \x01(\x0b\x32\x14.osi3.BaseStationary\x12\x45\n\x08geometry\x18\x04 \x01(\x0e\x32\x33.osi3.DetectedTrafficSign.DetectedMainSign.Geometry\x1ak\n\x11\x43\x61ndidateMainSign\x12\x13\n\x0bprobability\x18\x01 \x01(\x01\x12\x41\n\x0e\x63lassification\x18\x02 \x01(\x0b\x32).osi3.TrafficSign.MainSign.Classification\"\xb3\x02\n\x08Geometry\x12\x14\n\x10GEOMETRY_UNKNOWN\x10\x00\x12\x12\n\x0eGEOMETRY_OTHER\x10\x01\x12\x13\n\x0fGEOMETRY_CIRCLE\x10\x02\x12\x19\n\x15GEOMETRY_TRIANGLE_TOP\x10\x03\x12\x1a\n\x16GEOMETRY_TRIANGLE_DOWN\x10\x04\x12\x13\n\x0fGEOMETRY_SQUARE\x10\x05\x12\x11\n\rGEOMETRY_POLE\x10\x06\x12\x16\n\x12GEOMETRY_RECTANGLE\x10\x07\x12\x12\n\x0eGEOMETRY_PLATE\x10\x08\x12\x14\n\x10GEOMETRY_DIAMOND\x10\t\x12\x17\n\x13GEOMETRY_ARROW_LEFT\x10\n\x12\x18\n\x14GEOMETRY_ARROW_RIGHT\x10\x0b\x12\x14\n\x10GEOMETRY_OCTAGON\x10\x0c\x1a\xca\x02\n\x19\x44\x65tectedSupplementarySign\x12\x61\n\tcandidate\x18\x01 \x03(\x0b\x32N.osi3.DetectedTrafficSign.DetectedSupplementarySign.CandidateSupplementarySign\x12\"\n\x04\x62\x61se\x18\x02 \x01(\x0b\x32\x14.osi3.BaseStationary\x12\'\n\tbase_rmse\x18\x03 \x01(\x0b\x32\x14.osi3.BaseStationary\x1a}\n\x1a\x43\x61ndidateSupplementarySign\x12\x13\n\x0bprobability\x18\x01 \x01(\x01\x12J\n\x0e\x63lassification\x18\x02 \x01(\x0b\x32\x32.osi3.TrafficSign.SupplementarySign.ClassificationB\x02H\x01')
  ,
  dependencies=[osi_common_pb2.DESCRIPTOR,osi_trafficsign_pb2.DESCRIPTOR,osi_detectedobject_pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)



_DETECTEDTRAFFICSIGN_DETECTEDMAINSIGN_GEOMETRY = _descriptor.EnumDescriptor(
  name='Geometry',
  full_name='osi3.DetectedTrafficSign.DetectedMainSign.Geometry',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='GEOMETRY_UNKNOWN', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GEOMETRY_OTHER', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GEOMETRY_CIRCLE', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GEOMETRY_TRIANGLE_TOP', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GEOMETRY_TRIANGLE_DOWN', index=4, number=4,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GEOMETRY_SQUARE', index=5, number=5,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GEOMETRY_POLE', index=6, number=6,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GEOMETRY_RECTANGLE', index=7, number=7,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GEOMETRY_PLATE', index=8, number=8,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GEOMETRY_DIAMOND', index=9, number=9,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GEOMETRY_ARROW_LEFT', index=10, number=10,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GEOMETRY_ARROW_RIGHT', index=11, number=11,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GEOMETRY_OCTAGON', index=12, number=12,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=676,
  serialized_end=983,
)
_sym_db.RegisterEnumDescriptor(_DETECTEDTRAFFICSIGN_DETECTEDMAINSIGN_GEOMETRY)


_DETECTEDTRAFFICSIGN_DETECTEDMAINSIGN_CANDIDATEMAINSIGN = _descriptor.Descriptor(
  name='CandidateMainSign',
  full_name='osi3.DetectedTrafficSign.DetectedMainSign.CandidateMainSign',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='probability', full_name='osi3.DetectedTrafficSign.DetectedMainSign.CandidateMainSign.probability', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='classification', full_name='osi3.DetectedTrafficSign.DetectedMainSign.CandidateMainSign.classification', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=566,
  serialized_end=673,
)

_DETECTEDTRAFFICSIGN_DETECTEDMAINSIGN = _descriptor.Descriptor(
  name='DetectedMainSign',
  full_name='osi3.DetectedTrafficSign.DetectedMainSign',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='candidate', full_name='osi3.DetectedTrafficSign.DetectedMainSign.candidate', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='base', full_name='osi3.DetectedTrafficSign.DetectedMainSign.base', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='base_rmse', full_name='osi3.DetectedTrafficSign.DetectedMainSign.base_rmse', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='geometry', full_name='osi3.DetectedTrafficSign.DetectedMainSign.geometry', index=3,
      number=4, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[_DETECTEDTRAFFICSIGN_DETECTEDMAINSIGN_CANDIDATEMAINSIGN, ],
  enum_types=[
    _DETECTEDTRAFFICSIGN_DETECTEDMAINSIGN_GEOMETRY,
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=317,
  serialized_end=983,
)

_DETECTEDTRAFFICSIGN_DETECTEDSUPPLEMENTARYSIGN_CANDIDATESUPPLEMENTARYSIGN = _descriptor.Descriptor(
  name='CandidateSupplementarySign',
  full_name='osi3.DetectedTrafficSign.DetectedSupplementarySign.CandidateSupplementarySign',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='probability', full_name='osi3.DetectedTrafficSign.DetectedSupplementarySign.CandidateSupplementarySign.probability', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='classification', full_name='osi3.DetectedTrafficSign.DetectedSupplementarySign.CandidateSupplementarySign.classification', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1191,
  serialized_end=1316,
)

_DETECTEDTRAFFICSIGN_DETECTEDSUPPLEMENTARYSIGN = _descriptor.Descriptor(
  name='DetectedSupplementarySign',
  full_name='osi3.DetectedTrafficSign.DetectedSupplementarySign',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='candidate', full_name='osi3.DetectedTrafficSign.DetectedSupplementarySign.candidate', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='base', full_name='osi3.DetectedTrafficSign.DetectedSupplementarySign.base', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='base_rmse', full_name='osi3.DetectedTrafficSign.DetectedSupplementarySign.base_rmse', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[_DETECTEDTRAFFICSIGN_DETECTEDSUPPLEMENTARYSIGN_CANDIDATESUPPLEMENTARYSIGN, ],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=986,
  serialized_end=1316,
)

_DETECTEDTRAFFICSIGN = _descriptor.Descriptor(
  name='DetectedTrafficSign',
  full_name='osi3.DetectedTrafficSign',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='header', full_name='osi3.DetectedTrafficSign.header', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='main_sign', full_name='osi3.DetectedTrafficSign.main_sign', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='supplementary_sign', full_name='osi3.DetectedTrafficSign.supplementary_sign', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[_DETECTEDTRAFFICSIGN_DETECTEDMAINSIGN, _DETECTEDTRAFFICSIGN_DETECTEDSUPPLEMENTARYSIGN, ],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=107,
  serialized_end=1316,
)

_DETECTEDTRAFFICSIGN_DETECTEDMAINSIGN_CANDIDATEMAINSIGN.fields_by_name['classification'].message_type = osi_trafficsign_pb2._TRAFFICSIGN_MAINSIGN_CLASSIFICATION
_DETECTEDTRAFFICSIGN_DETECTEDMAINSIGN_CANDIDATEMAINSIGN.containing_type = _DETECTEDTRAFFICSIGN_DETECTEDMAINSIGN
_DETECTEDTRAFFICSIGN_DETECTEDMAINSIGN.fields_by_name['candidate'].message_type = _DETECTEDTRAFFICSIGN_DETECTEDMAINSIGN_CANDIDATEMAINSIGN
_DETECTEDTRAFFICSIGN_DETECTEDMAINSIGN.fields_by_name['base'].message_type = osi_common_pb2._BASESTATIONARY
_DETECTEDTRAFFICSIGN_DETECTEDMAINSIGN.fields_by_name['base_rmse'].message_type = osi_common_pb2._BASESTATIONARY
_DETECTEDTRAFFICSIGN_DETECTEDMAINSIGN.fields_by_name['geometry'].enum_type = _DETECTEDTRAFFICSIGN_DETECTEDMAINSIGN_GEOMETRY
_DETECTEDTRAFFICSIGN_DETECTEDMAINSIGN.containing_type = _DETECTEDTRAFFICSIGN
_DETECTEDTRAFFICSIGN_DETECTEDMAINSIGN_GEOMETRY.containing_type = _DETECTEDTRAFFICSIGN_DETECTEDMAINSIGN
_DETECTEDTRAFFICSIGN_DETECTEDSUPPLEMENTARYSIGN_CANDIDATESUPPLEMENTARYSIGN.fields_by_name['classification'].message_type = osi_trafficsign_pb2._TRAFFICSIGN_SUPPLEMENTARYSIGN_CLASSIFICATION
_DETECTEDTRAFFICSIGN_DETECTEDSUPPLEMENTARYSIGN_CANDIDATESUPPLEMENTARYSIGN.containing_type = _DETECTEDTRAFFICSIGN_DETECTEDSUPPLEMENTARYSIGN
_DETECTEDTRAFFICSIGN_DETECTEDSUPPLEMENTARYSIGN.fields_by_name['candidate'].message_type = _DETECTEDTRAFFICSIGN_DETECTEDSUPPLEMENTARYSIGN_CANDIDATESUPPLEMENTARYSIGN
_DETECTEDTRAFFICSIGN_DETECTEDSUPPLEMENTARYSIGN.fields_by_name['base'].message_type = osi_common_pb2._BASESTATIONARY
_DETECTEDTRAFFICSIGN_DETECTEDSUPPLEMENTARYSIGN.fields_by_name['base_rmse'].message_type = osi_common_pb2._BASESTATIONARY
_DETECTEDTRAFFICSIGN_DETECTEDSUPPLEMENTARYSIGN.containing_type = _DETECTEDTRAFFICSIGN
_DETECTEDTRAFFICSIGN.fields_by_name['header'].message_type = osi_detectedobject_pb2._DETECTEDITEMHEADER
_DETECTEDTRAFFICSIGN.fields_by_name['main_sign'].message_type = _DETECTEDTRAFFICSIGN_DETECTEDMAINSIGN
_DETECTEDTRAFFICSIGN.fields_by_name['supplementary_sign'].message_type = _DETECTEDTRAFFICSIGN_DETECTEDSUPPLEMENTARYSIGN
DESCRIPTOR.message_types_by_name['DetectedTrafficSign'] = _DETECTEDTRAFFICSIGN

DetectedTrafficSign = _reflection.GeneratedProtocolMessageType('DetectedTrafficSign', (_message.Message,), dict(

  DetectedMainSign = _reflection.GeneratedProtocolMessageType('DetectedMainSign', (_message.Message,), dict(

    CandidateMainSign = _reflection.GeneratedProtocolMessageType('CandidateMainSign', (_message.Message,), dict(
      DESCRIPTOR = _DETECTEDTRAFFICSIGN_DETECTEDMAINSIGN_CANDIDATEMAINSIGN,
      __module__ = 'osi_detectedtrafficsign_pb2'
      # @@protoc_insertion_point(class_scope:osi3.DetectedTrafficSign.DetectedMainSign.CandidateMainSign)
      ))
    ,
    DESCRIPTOR = _DETECTEDTRAFFICSIGN_DETECTEDMAINSIGN,
    __module__ = 'osi_detectedtrafficsign_pb2'
    # @@protoc_insertion_point(class_scope:osi3.DetectedTrafficSign.DetectedMainSign)
    ))
  ,

  DetectedSupplementarySign = _reflection.GeneratedProtocolMessageType('DetectedSupplementarySign', (_message.Message,), dict(

    CandidateSupplementarySign = _reflection.GeneratedProtocolMessageType('CandidateSupplementarySign', (_message.Message,), dict(
      DESCRIPTOR = _DETECTEDTRAFFICSIGN_DETECTEDSUPPLEMENTARYSIGN_CANDIDATESUPPLEMENTARYSIGN,
      __module__ = 'osi_detectedtrafficsign_pb2'
      # @@protoc_insertion_point(class_scope:osi3.DetectedTrafficSign.DetectedSupplementarySign.CandidateSupplementarySign)
      ))
    ,
    DESCRIPTOR = _DETECTEDTRAFFICSIGN_DETECTEDSUPPLEMENTARYSIGN,
    __module__ = 'osi_detectedtrafficsign_pb2'
    # @@protoc_insertion_point(class_scope:osi3.DetectedTrafficSign.DetectedSupplementarySign)
    ))
  ,
  DESCRIPTOR = _DETECTEDTRAFFICSIGN,
  __module__ = 'osi_detectedtrafficsign_pb2'
  # @@protoc_insertion_point(class_scope:osi3.DetectedTrafficSign)
  ))
_sym_db.RegisterMessage(DetectedTrafficSign)
_sym_db.RegisterMessage(DetectedTrafficSign.DetectedMainSign)
_sym_db.RegisterMessage(DetectedTrafficSign.DetectedMainSign.CandidateMainSign)
_sym_db.RegisterMessage(DetectedTrafficSign.DetectedSupplementarySign)
_sym_db.RegisterMessage(DetectedTrafficSign.DetectedSupplementarySign.CandidateSupplementarySign)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('H\001'))
# @@protoc_insertion_point(module_scope)
