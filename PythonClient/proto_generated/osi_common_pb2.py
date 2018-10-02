# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: osi_common.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='osi_common.proto',
  package='osi3',
  serialized_pb=_b('\n\x10osi_common.proto\x12\x04osi3\"+\n\x08Vector3d\x12\t\n\x01x\x18\x01 \x01(\x01\x12\t\n\x01y\x18\x02 \x01(\x01\x12\t\n\x01z\x18\x03 \x01(\x01\" \n\x08Vector2d\x12\t\n\x01x\x18\x01 \x01(\x01\x12\t\n\x01y\x18\x02 \x01(\x01\"+\n\tTimestamp\x12\x0f\n\x07seconds\x18\x01 \x01(\x03\x12\r\n\x05nanos\x18\x02 \x01(\r\"<\n\x0b\x44imension3d\x12\x0e\n\x06length\x18\x01 \x01(\x01\x12\r\n\x05width\x18\x02 \x01(\x01\x12\x0e\n\x06height\x18\x03 \x01(\x01\"9\n\rOrientation3d\x12\x0c\n\x04roll\x18\x01 \x01(\x01\x12\r\n\x05pitch\x18\x02 \x01(\x01\x12\x0b\n\x03yaw\x18\x03 \x01(\x01\"\x1b\n\nIdentifier\x12\r\n\x05value\x18\x01 \x01(\x04\"^\n\x10MountingPosition\x12 \n\x08position\x18\x01 \x01(\x0b\x32\x0e.osi3.Vector3d\x12(\n\x0borientation\x18\x02 \x01(\x0b\x32\x13.osi3.Orientation3d\"C\n\x0bSpherical3d\x12\x10\n\x08\x64istance\x18\x01 \x01(\x01\x12\x0f\n\x07\x61zimuth\x18\x02 \x01(\x01\x12\x11\n\televation\x18\x03 \x01(\x01\"\xa8\x01\n\x0e\x42\x61seStationary\x12$\n\tdimension\x18\x01 \x01(\x0b\x32\x11.osi3.Dimension3d\x12 \n\x08position\x18\x02 \x01(\x0b\x32\x0e.osi3.Vector3d\x12(\n\x0borientation\x18\x03 \x01(\x0b\x32\x13.osi3.Orientation3d\x12$\n\x0c\x62\x61se_polygon\x18\x04 \x03(\x0b\x32\x0e.osi3.Vector2d\"\x9b\x02\n\nBaseMoving\x12$\n\tdimension\x18\x01 \x01(\x0b\x32\x11.osi3.Dimension3d\x12 \n\x08position\x18\x02 \x01(\x0b\x32\x0e.osi3.Vector3d\x12(\n\x0borientation\x18\x03 \x01(\x0b\x32\x13.osi3.Orientation3d\x12 \n\x08velocity\x18\x04 \x01(\x0b\x32\x0e.osi3.Vector3d\x12$\n\x0c\x61\x63\x63\x65leration\x18\x05 \x01(\x0b\x32\x0e.osi3.Vector3d\x12-\n\x10orientation_rate\x18\x06 \x01(\x0b\x32\x13.osi3.Orientation3d\x12$\n\x0c\x62\x61se_polygon\x18\x07 \x03(\x0b\x32\x0e.osi3.Vector2dB\x02H\x01')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_VECTOR3D = _descriptor.Descriptor(
  name='Vector3d',
  full_name='osi3.Vector3d',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='osi3.Vector3d.x', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='y', full_name='osi3.Vector3d.y', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='z', full_name='osi3.Vector3d.z', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=26,
  serialized_end=69,
)


_VECTOR2D = _descriptor.Descriptor(
  name='Vector2d',
  full_name='osi3.Vector2d',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='osi3.Vector2d.x', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='y', full_name='osi3.Vector2d.y', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=71,
  serialized_end=103,
)


_TIMESTAMP = _descriptor.Descriptor(
  name='Timestamp',
  full_name='osi3.Timestamp',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='seconds', full_name='osi3.Timestamp.seconds', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='nanos', full_name='osi3.Timestamp.nanos', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=105,
  serialized_end=148,
)


_DIMENSION3D = _descriptor.Descriptor(
  name='Dimension3d',
  full_name='osi3.Dimension3d',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='length', full_name='osi3.Dimension3d.length', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='width', full_name='osi3.Dimension3d.width', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='height', full_name='osi3.Dimension3d.height', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=150,
  serialized_end=210,
)


_ORIENTATION3D = _descriptor.Descriptor(
  name='Orientation3d',
  full_name='osi3.Orientation3d',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='roll', full_name='osi3.Orientation3d.roll', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='pitch', full_name='osi3.Orientation3d.pitch', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='yaw', full_name='osi3.Orientation3d.yaw', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=212,
  serialized_end=269,
)


_IDENTIFIER = _descriptor.Descriptor(
  name='Identifier',
  full_name='osi3.Identifier',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='osi3.Identifier.value', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=271,
  serialized_end=298,
)


_MOUNTINGPOSITION = _descriptor.Descriptor(
  name='MountingPosition',
  full_name='osi3.MountingPosition',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='position', full_name='osi3.MountingPosition.position', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='orientation', full_name='osi3.MountingPosition.orientation', index=1,
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
  serialized_start=300,
  serialized_end=394,
)


_SPHERICAL3D = _descriptor.Descriptor(
  name='Spherical3d',
  full_name='osi3.Spherical3d',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='distance', full_name='osi3.Spherical3d.distance', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='azimuth', full_name='osi3.Spherical3d.azimuth', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='elevation', full_name='osi3.Spherical3d.elevation', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=396,
  serialized_end=463,
)


_BASESTATIONARY = _descriptor.Descriptor(
  name='BaseStationary',
  full_name='osi3.BaseStationary',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='dimension', full_name='osi3.BaseStationary.dimension', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='position', full_name='osi3.BaseStationary.position', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='orientation', full_name='osi3.BaseStationary.orientation', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='base_polygon', full_name='osi3.BaseStationary.base_polygon', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=466,
  serialized_end=634,
)


_BASEMOVING = _descriptor.Descriptor(
  name='BaseMoving',
  full_name='osi3.BaseMoving',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='dimension', full_name='osi3.BaseMoving.dimension', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='position', full_name='osi3.BaseMoving.position', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='orientation', full_name='osi3.BaseMoving.orientation', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='velocity', full_name='osi3.BaseMoving.velocity', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='acceleration', full_name='osi3.BaseMoving.acceleration', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='orientation_rate', full_name='osi3.BaseMoving.orientation_rate', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='base_polygon', full_name='osi3.BaseMoving.base_polygon', index=6,
      number=7, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=637,
  serialized_end=920,
)

_MOUNTINGPOSITION.fields_by_name['position'].message_type = _VECTOR3D
_MOUNTINGPOSITION.fields_by_name['orientation'].message_type = _ORIENTATION3D
_BASESTATIONARY.fields_by_name['dimension'].message_type = _DIMENSION3D
_BASESTATIONARY.fields_by_name['position'].message_type = _VECTOR3D
_BASESTATIONARY.fields_by_name['orientation'].message_type = _ORIENTATION3D
_BASESTATIONARY.fields_by_name['base_polygon'].message_type = _VECTOR2D
_BASEMOVING.fields_by_name['dimension'].message_type = _DIMENSION3D
_BASEMOVING.fields_by_name['position'].message_type = _VECTOR3D
_BASEMOVING.fields_by_name['orientation'].message_type = _ORIENTATION3D
_BASEMOVING.fields_by_name['velocity'].message_type = _VECTOR3D
_BASEMOVING.fields_by_name['acceleration'].message_type = _VECTOR3D
_BASEMOVING.fields_by_name['orientation_rate'].message_type = _ORIENTATION3D
_BASEMOVING.fields_by_name['base_polygon'].message_type = _VECTOR2D
DESCRIPTOR.message_types_by_name['Vector3d'] = _VECTOR3D
DESCRIPTOR.message_types_by_name['Vector2d'] = _VECTOR2D
DESCRIPTOR.message_types_by_name['Timestamp'] = _TIMESTAMP
DESCRIPTOR.message_types_by_name['Dimension3d'] = _DIMENSION3D
DESCRIPTOR.message_types_by_name['Orientation3d'] = _ORIENTATION3D
DESCRIPTOR.message_types_by_name['Identifier'] = _IDENTIFIER
DESCRIPTOR.message_types_by_name['MountingPosition'] = _MOUNTINGPOSITION
DESCRIPTOR.message_types_by_name['Spherical3d'] = _SPHERICAL3D
DESCRIPTOR.message_types_by_name['BaseStationary'] = _BASESTATIONARY
DESCRIPTOR.message_types_by_name['BaseMoving'] = _BASEMOVING

Vector3d = _reflection.GeneratedProtocolMessageType('Vector3d', (_message.Message,), dict(
  DESCRIPTOR = _VECTOR3D,
  __module__ = 'osi_common_pb2'
  # @@protoc_insertion_point(class_scope:osi3.Vector3d)
  ))
_sym_db.RegisterMessage(Vector3d)

Vector2d = _reflection.GeneratedProtocolMessageType('Vector2d', (_message.Message,), dict(
  DESCRIPTOR = _VECTOR2D,
  __module__ = 'osi_common_pb2'
  # @@protoc_insertion_point(class_scope:osi3.Vector2d)
  ))
_sym_db.RegisterMessage(Vector2d)

Timestamp = _reflection.GeneratedProtocolMessageType('Timestamp', (_message.Message,), dict(
  DESCRIPTOR = _TIMESTAMP,
  __module__ = 'osi_common_pb2'
  # @@protoc_insertion_point(class_scope:osi3.Timestamp)
  ))
_sym_db.RegisterMessage(Timestamp)

Dimension3d = _reflection.GeneratedProtocolMessageType('Dimension3d', (_message.Message,), dict(
  DESCRIPTOR = _DIMENSION3D,
  __module__ = 'osi_common_pb2'
  # @@protoc_insertion_point(class_scope:osi3.Dimension3d)
  ))
_sym_db.RegisterMessage(Dimension3d)

Orientation3d = _reflection.GeneratedProtocolMessageType('Orientation3d', (_message.Message,), dict(
  DESCRIPTOR = _ORIENTATION3D,
  __module__ = 'osi_common_pb2'
  # @@protoc_insertion_point(class_scope:osi3.Orientation3d)
  ))
_sym_db.RegisterMessage(Orientation3d)

Identifier = _reflection.GeneratedProtocolMessageType('Identifier', (_message.Message,), dict(
  DESCRIPTOR = _IDENTIFIER,
  __module__ = 'osi_common_pb2'
  # @@protoc_insertion_point(class_scope:osi3.Identifier)
  ))
_sym_db.RegisterMessage(Identifier)

MountingPosition = _reflection.GeneratedProtocolMessageType('MountingPosition', (_message.Message,), dict(
  DESCRIPTOR = _MOUNTINGPOSITION,
  __module__ = 'osi_common_pb2'
  # @@protoc_insertion_point(class_scope:osi3.MountingPosition)
  ))
_sym_db.RegisterMessage(MountingPosition)

Spherical3d = _reflection.GeneratedProtocolMessageType('Spherical3d', (_message.Message,), dict(
  DESCRIPTOR = _SPHERICAL3D,
  __module__ = 'osi_common_pb2'
  # @@protoc_insertion_point(class_scope:osi3.Spherical3d)
  ))
_sym_db.RegisterMessage(Spherical3d)

BaseStationary = _reflection.GeneratedProtocolMessageType('BaseStationary', (_message.Message,), dict(
  DESCRIPTOR = _BASESTATIONARY,
  __module__ = 'osi_common_pb2'
  # @@protoc_insertion_point(class_scope:osi3.BaseStationary)
  ))
_sym_db.RegisterMessage(BaseStationary)

BaseMoving = _reflection.GeneratedProtocolMessageType('BaseMoving', (_message.Message,), dict(
  DESCRIPTOR = _BASEMOVING,
  __module__ = 'osi_common_pb2'
  # @@protoc_insertion_point(class_scope:osi3.BaseMoving)
  ))
_sym_db.RegisterMessage(BaseMoving)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('H\001'))
# @@protoc_insertion_point(module_scope)
