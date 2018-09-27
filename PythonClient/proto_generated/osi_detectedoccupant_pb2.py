# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: osi_detectedoccupant.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import osi_occupant_pb2
import osi_detectedobject_pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='osi_detectedoccupant.proto',
  package='osi3',
  serialized_pb=_b('\n\x1aosi_detectedoccupant.proto\x12\x04osi3\x1a\x12osi_occupant.proto\x1a\x18osi_detectedobject.proto\"\xda\x01\n\x10\x44\x65tectedOccupant\x12(\n\x06header\x18\x01 \x01(\x0b\x32\x18.osi3.DetectedItemHeader\x12;\n\tcandidate\x18\x02 \x03(\x0b\x32(.osi3.DetectedOccupant.CandidateOccupant\x1a_\n\x11\x43\x61ndidateOccupant\x12\x13\n\x0bprobability\x18\x01 \x01(\x01\x12\x35\n\x0e\x63lassification\x18\x02 \x01(\x0b\x32\x1d.osi3.Occupant.ClassificationB\x02H\x01')
  ,
  dependencies=[osi_occupant_pb2.DESCRIPTOR,osi_detectedobject_pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_DETECTEDOCCUPANT_CANDIDATEOCCUPANT = _descriptor.Descriptor(
  name='CandidateOccupant',
  full_name='osi3.DetectedOccupant.CandidateOccupant',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='probability', full_name='osi3.DetectedOccupant.CandidateOccupant.probability', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='classification', full_name='osi3.DetectedOccupant.CandidateOccupant.classification', index=1,
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
  serialized_start=206,
  serialized_end=301,
)

_DETECTEDOCCUPANT = _descriptor.Descriptor(
  name='DetectedOccupant',
  full_name='osi3.DetectedOccupant',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='header', full_name='osi3.DetectedOccupant.header', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='candidate', full_name='osi3.DetectedOccupant.candidate', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[_DETECTEDOCCUPANT_CANDIDATEOCCUPANT, ],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=83,
  serialized_end=301,
)

_DETECTEDOCCUPANT_CANDIDATEOCCUPANT.fields_by_name['classification'].message_type = osi_occupant_pb2._OCCUPANT_CLASSIFICATION
_DETECTEDOCCUPANT_CANDIDATEOCCUPANT.containing_type = _DETECTEDOCCUPANT
_DETECTEDOCCUPANT.fields_by_name['header'].message_type = osi_detectedobject_pb2._DETECTEDITEMHEADER
_DETECTEDOCCUPANT.fields_by_name['candidate'].message_type = _DETECTEDOCCUPANT_CANDIDATEOCCUPANT
DESCRIPTOR.message_types_by_name['DetectedOccupant'] = _DETECTEDOCCUPANT

DetectedOccupant = _reflection.GeneratedProtocolMessageType('DetectedOccupant', (_message.Message,), dict(

  CandidateOccupant = _reflection.GeneratedProtocolMessageType('CandidateOccupant', (_message.Message,), dict(
    DESCRIPTOR = _DETECTEDOCCUPANT_CANDIDATEOCCUPANT,
    __module__ = 'osi_detectedoccupant_pb2'
    # @@protoc_insertion_point(class_scope:osi3.DetectedOccupant.CandidateOccupant)
    ))
  ,
  DESCRIPTOR = _DETECTEDOCCUPANT,
  __module__ = 'osi_detectedoccupant_pb2'
  # @@protoc_insertion_point(class_scope:osi3.DetectedOccupant)
  ))
_sym_db.RegisterMessage(DetectedOccupant)
_sym_db.RegisterMessage(DetectedOccupant.CandidateOccupant)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('H\001'))
# @@protoc_insertion_point(module_scope)
