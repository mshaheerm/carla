# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: osi_hostvehicledata.proto

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


DESCRIPTOR = _descriptor.FileDescriptor(
  name='osi_hostvehicledata.proto',
  package='osi3',
  serialized_pb=_b('\n\x19osi_hostvehicledata.proto\x12\x04osi3\x1a\x10osi_common.proto\"^\n\x0fHostVehicleData\x12\"\n\x08location\x18\x01 \x01(\x0b\x32\x10.osi3.BaseMoving\x12\'\n\rlocation_rmse\x18\x02 \x01(\x0b\x32\x10.osi3.BaseMovingB\x02H\x01')
  ,
  dependencies=[osi_common_pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_HOSTVEHICLEDATA = _descriptor.Descriptor(
  name='HostVehicleData',
  full_name='osi3.HostVehicleData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='location', full_name='osi3.HostVehicleData.location', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='location_rmse', full_name='osi3.HostVehicleData.location_rmse', index=1,
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
  serialized_start=53,
  serialized_end=147,
)

_HOSTVEHICLEDATA.fields_by_name['location'].message_type = osi_common_pb2._BASEMOVING
_HOSTVEHICLEDATA.fields_by_name['location_rmse'].message_type = osi_common_pb2._BASEMOVING
DESCRIPTOR.message_types_by_name['HostVehicleData'] = _HOSTVEHICLEDATA

HostVehicleData = _reflection.GeneratedProtocolMessageType('HostVehicleData', (_message.Message,), dict(
  DESCRIPTOR = _HOSTVEHICLEDATA,
  __module__ = 'osi_hostvehicledata_pb2'
  # @@protoc_insertion_point(class_scope:osi3.HostVehicleData)
  ))
_sym_db.RegisterMessage(HostVehicleData)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('H\001'))
# @@protoc_insertion_point(module_scope)
