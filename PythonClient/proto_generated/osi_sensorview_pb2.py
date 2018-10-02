# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: osi_sensorview.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import osi_version_pb2
import osi_common_pb2
import osi_groundtruth_pb2
import osi_sensorviewconfiguration_pb2
import osi_hostvehicledata_pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='osi_sensorview.proto',
  package='osi3',
  serialized_pb=_b('\n\x14osi_sensorview.proto\x12\x04osi3\x1a\x11osi_version.proto\x1a\x10osi_common.proto\x1a\x15osi_groundtruth.proto\x1a!osi_sensorviewconfiguration.proto\x1a\x19osi_hostvehicledata.proto\"\x85\x05\n\nSensorView\x12\'\n\x07version\x18\x01 \x01(\x0b\x32\x16.osi3.InterfaceVersion\x12\"\n\ttimestamp\x18\x02 \x01(\x0b\x32\x0f.osi3.Timestamp\x12#\n\tsensor_id\x18\x03 \x01(\x0b\x32\x10.osi3.Identifier\x12\x31\n\x11mounting_position\x18\x04 \x01(\x0b\x32\x16.osi3.MountingPosition\x12\x36\n\x16mounting_position_rmse\x18\x05 \x01(\x0b\x32\x16.osi3.MountingPosition\x12\x30\n\x11host_vehicle_data\x18\x06 \x01(\x0b\x32\x15.osi3.HostVehicleData\x12.\n\x13global_ground_truth\x18\x07 \x01(\x0b\x32\x11.osi3.GroundTruth\x12)\n\x0fhost_vehicle_id\x18\x08 \x01(\x0b\x32\x10.osi3.Identifier\x12\x35\n\x13generic_sensor_view\x18\xe8\x07 \x03(\x0b\x32\x17.osi3.GenericSensorView\x12\x31\n\x11radar_sensor_view\x18\xe9\x07 \x03(\x0b\x32\x15.osi3.RadarSensorView\x12\x31\n\x11lidar_sensor_view\x18\xea\x07 \x03(\x0b\x32\x15.osi3.LidarSensorView\x12\x33\n\x12\x63\x61mera_sensor_view\x18\xeb\x07 \x03(\x0b\x32\x16.osi3.CameraSensorView\x12;\n\x16ultrasonic_sensor_view\x18\xec\x07 \x03(\x0b\x32\x1a.osi3.UltrasonicSensorView\"U\n\x11GenericSensorView\x12@\n\x12view_configuration\x18\x01 \x01(\x0b\x32$.osi3.GenericSensorViewConfiguration\"\x9e\x02\n\x0fRadarSensorView\x12>\n\x12view_configuration\x18\x01 \x01(\x0b\x32\".osi3.RadarSensorViewConfiguration\x12\x34\n\nreflection\x18\x02 \x03(\x0b\x32 .osi3.RadarSensorView.Reflection\x1a\x94\x01\n\nReflection\x12\x17\n\x0fsignal_strength\x18\x01 \x01(\x01\x12\x16\n\x0etime_of_flight\x18\x02 \x01(\x01\x12\x15\n\rdoppler_shift\x18\x03 \x01(\x01\x12\x1f\n\x17source_horizontal_angle\x18\x04 \x01(\x01\x12\x1d\n\x15source_vertical_angle\x18\x05 \x01(\x01\"\xdd\x01\n\x0fLidarSensorView\x12>\n\x12view_configuration\x18\x01 \x01(\x0b\x32\".osi3.LidarSensorViewConfiguration\x12\x34\n\nreflection\x18\x02 \x03(\x0b\x32 .osi3.LidarSensorView.Reflection\x1aT\n\nReflection\x12\x17\n\x0fsignal_strength\x18\x01 \x01(\x01\x12\x16\n\x0etime_of_flight\x18\x02 \x01(\x01\x12\x15\n\rdoppler_shift\x18\x03 \x01(\x01\"g\n\x10\x43\x61meraSensorView\x12?\n\x12view_configuration\x18\x01 \x01(\x0b\x32#.osi3.CameraSensorViewConfiguration\x12\x12\n\nimage_data\x18\x02 \x01(\x0c\"[\n\x14UltrasonicSensorView\x12\x43\n\x12view_configuration\x18\x01 \x01(\x0b\x32\'.osi3.UltrasonicSensorViewConfigurationB\x02H\x01')
  ,
  dependencies=[osi_version_pb2.DESCRIPTOR,osi_common_pb2.DESCRIPTOR,osi_groundtruth_pb2.DESCRIPTOR,osi_sensorviewconfiguration_pb2.DESCRIPTOR,osi_hostvehicledata_pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_SENSORVIEW = _descriptor.Descriptor(
  name='SensorView',
  full_name='osi3.SensorView',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='version', full_name='osi3.SensorView.version', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='timestamp', full_name='osi3.SensorView.timestamp', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='sensor_id', full_name='osi3.SensorView.sensor_id', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='mounting_position', full_name='osi3.SensorView.mounting_position', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='mounting_position_rmse', full_name='osi3.SensorView.mounting_position_rmse', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='host_vehicle_data', full_name='osi3.SensorView.host_vehicle_data', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='global_ground_truth', full_name='osi3.SensorView.global_ground_truth', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='host_vehicle_id', full_name='osi3.SensorView.host_vehicle_id', index=7,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='generic_sensor_view', full_name='osi3.SensorView.generic_sensor_view', index=8,
      number=1000, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='radar_sensor_view', full_name='osi3.SensorView.radar_sensor_view', index=9,
      number=1001, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='lidar_sensor_view', full_name='osi3.SensorView.lidar_sensor_view', index=10,
      number=1002, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='camera_sensor_view', full_name='osi3.SensorView.camera_sensor_view', index=11,
      number=1003, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='ultrasonic_sensor_view', full_name='osi3.SensorView.ultrasonic_sensor_view', index=12,
      number=1004, type=11, cpp_type=10, label=3,
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
  serialized_start=153,
  serialized_end=798,
)


_GENERICSENSORVIEW = _descriptor.Descriptor(
  name='GenericSensorView',
  full_name='osi3.GenericSensorView',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='view_configuration', full_name='osi3.GenericSensorView.view_configuration', index=0,
      number=1, type=11, cpp_type=10, label=1,
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
  serialized_start=800,
  serialized_end=885,
)


_RADARSENSORVIEW_REFLECTION = _descriptor.Descriptor(
  name='Reflection',
  full_name='osi3.RadarSensorView.Reflection',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='signal_strength', full_name='osi3.RadarSensorView.Reflection.signal_strength', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='time_of_flight', full_name='osi3.RadarSensorView.Reflection.time_of_flight', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='doppler_shift', full_name='osi3.RadarSensorView.Reflection.doppler_shift', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='source_horizontal_angle', full_name='osi3.RadarSensorView.Reflection.source_horizontal_angle', index=3,
      number=4, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='source_vertical_angle', full_name='osi3.RadarSensorView.Reflection.source_vertical_angle', index=4,
      number=5, type=1, cpp_type=5, label=1,
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
  serialized_start=1026,
  serialized_end=1174,
)

_RADARSENSORVIEW = _descriptor.Descriptor(
  name='RadarSensorView',
  full_name='osi3.RadarSensorView',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='view_configuration', full_name='osi3.RadarSensorView.view_configuration', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='reflection', full_name='osi3.RadarSensorView.reflection', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[_RADARSENSORVIEW_REFLECTION, ],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=888,
  serialized_end=1174,
)


_LIDARSENSORVIEW_REFLECTION = _descriptor.Descriptor(
  name='Reflection',
  full_name='osi3.LidarSensorView.Reflection',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='signal_strength', full_name='osi3.LidarSensorView.Reflection.signal_strength', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='time_of_flight', full_name='osi3.LidarSensorView.Reflection.time_of_flight', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='doppler_shift', full_name='osi3.LidarSensorView.Reflection.doppler_shift', index=2,
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
  serialized_start=1026,
  serialized_end=1110,
)

_LIDARSENSORVIEW = _descriptor.Descriptor(
  name='LidarSensorView',
  full_name='osi3.LidarSensorView',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='view_configuration', full_name='osi3.LidarSensorView.view_configuration', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='reflection', full_name='osi3.LidarSensorView.reflection', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[_LIDARSENSORVIEW_REFLECTION, ],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1177,
  serialized_end=1398,
)


_CAMERASENSORVIEW = _descriptor.Descriptor(
  name='CameraSensorView',
  full_name='osi3.CameraSensorView',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='view_configuration', full_name='osi3.CameraSensorView.view_configuration', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='image_data', full_name='osi3.CameraSensorView.image_data', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
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
  serialized_start=1400,
  serialized_end=1503,
)


_ULTRASONICSENSORVIEW = _descriptor.Descriptor(
  name='UltrasonicSensorView',
  full_name='osi3.UltrasonicSensorView',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='view_configuration', full_name='osi3.UltrasonicSensorView.view_configuration', index=0,
      number=1, type=11, cpp_type=10, label=1,
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
  serialized_start=1505,
  serialized_end=1596,
)

_SENSORVIEW.fields_by_name['version'].message_type = osi_version_pb2._INTERFACEVERSION
_SENSORVIEW.fields_by_name['timestamp'].message_type = osi_common_pb2._TIMESTAMP
_SENSORVIEW.fields_by_name['sensor_id'].message_type = osi_common_pb2._IDENTIFIER
_SENSORVIEW.fields_by_name['mounting_position'].message_type = osi_common_pb2._MOUNTINGPOSITION
_SENSORVIEW.fields_by_name['mounting_position_rmse'].message_type = osi_common_pb2._MOUNTINGPOSITION
_SENSORVIEW.fields_by_name['host_vehicle_data'].message_type = osi_hostvehicledata_pb2._HOSTVEHICLEDATA
_SENSORVIEW.fields_by_name['global_ground_truth'].message_type = osi_groundtruth_pb2._GROUNDTRUTH
_SENSORVIEW.fields_by_name['host_vehicle_id'].message_type = osi_common_pb2._IDENTIFIER
_SENSORVIEW.fields_by_name['generic_sensor_view'].message_type = _GENERICSENSORVIEW
_SENSORVIEW.fields_by_name['radar_sensor_view'].message_type = _RADARSENSORVIEW
_SENSORVIEW.fields_by_name['lidar_sensor_view'].message_type = _LIDARSENSORVIEW
_SENSORVIEW.fields_by_name['camera_sensor_view'].message_type = _CAMERASENSORVIEW
_SENSORVIEW.fields_by_name['ultrasonic_sensor_view'].message_type = _ULTRASONICSENSORVIEW
_GENERICSENSORVIEW.fields_by_name['view_configuration'].message_type = osi_sensorviewconfiguration_pb2._GENERICSENSORVIEWCONFIGURATION
_RADARSENSORVIEW_REFLECTION.containing_type = _RADARSENSORVIEW
_RADARSENSORVIEW.fields_by_name['view_configuration'].message_type = osi_sensorviewconfiguration_pb2._RADARSENSORVIEWCONFIGURATION
_RADARSENSORVIEW.fields_by_name['reflection'].message_type = _RADARSENSORVIEW_REFLECTION
_LIDARSENSORVIEW_REFLECTION.containing_type = _LIDARSENSORVIEW
_LIDARSENSORVIEW.fields_by_name['view_configuration'].message_type = osi_sensorviewconfiguration_pb2._LIDARSENSORVIEWCONFIGURATION
_LIDARSENSORVIEW.fields_by_name['reflection'].message_type = _LIDARSENSORVIEW_REFLECTION
_CAMERASENSORVIEW.fields_by_name['view_configuration'].message_type = osi_sensorviewconfiguration_pb2._CAMERASENSORVIEWCONFIGURATION
_ULTRASONICSENSORVIEW.fields_by_name['view_configuration'].message_type = osi_sensorviewconfiguration_pb2._ULTRASONICSENSORVIEWCONFIGURATION
DESCRIPTOR.message_types_by_name['SensorView'] = _SENSORVIEW
DESCRIPTOR.message_types_by_name['GenericSensorView'] = _GENERICSENSORVIEW
DESCRIPTOR.message_types_by_name['RadarSensorView'] = _RADARSENSORVIEW
DESCRIPTOR.message_types_by_name['LidarSensorView'] = _LIDARSENSORVIEW
DESCRIPTOR.message_types_by_name['CameraSensorView'] = _CAMERASENSORVIEW
DESCRIPTOR.message_types_by_name['UltrasonicSensorView'] = _ULTRASONICSENSORVIEW

SensorView = _reflection.GeneratedProtocolMessageType('SensorView', (_message.Message,), dict(
  DESCRIPTOR = _SENSORVIEW,
  __module__ = 'osi_sensorview_pb2'
  # @@protoc_insertion_point(class_scope:osi3.SensorView)
  ))
_sym_db.RegisterMessage(SensorView)

GenericSensorView = _reflection.GeneratedProtocolMessageType('GenericSensorView', (_message.Message,), dict(
  DESCRIPTOR = _GENERICSENSORVIEW,
  __module__ = 'osi_sensorview_pb2'
  # @@protoc_insertion_point(class_scope:osi3.GenericSensorView)
  ))
_sym_db.RegisterMessage(GenericSensorView)

RadarSensorView = _reflection.GeneratedProtocolMessageType('RadarSensorView', (_message.Message,), dict(

  Reflection = _reflection.GeneratedProtocolMessageType('Reflection', (_message.Message,), dict(
    DESCRIPTOR = _RADARSENSORVIEW_REFLECTION,
    __module__ = 'osi_sensorview_pb2'
    # @@protoc_insertion_point(class_scope:osi3.RadarSensorView.Reflection)
    ))
  ,
  DESCRIPTOR = _RADARSENSORVIEW,
  __module__ = 'osi_sensorview_pb2'
  # @@protoc_insertion_point(class_scope:osi3.RadarSensorView)
  ))
_sym_db.RegisterMessage(RadarSensorView)
_sym_db.RegisterMessage(RadarSensorView.Reflection)

LidarSensorView = _reflection.GeneratedProtocolMessageType('LidarSensorView', (_message.Message,), dict(

  Reflection = _reflection.GeneratedProtocolMessageType('Reflection', (_message.Message,), dict(
    DESCRIPTOR = _LIDARSENSORVIEW_REFLECTION,
    __module__ = 'osi_sensorview_pb2'
    # @@protoc_insertion_point(class_scope:osi3.LidarSensorView.Reflection)
    ))
  ,
  DESCRIPTOR = _LIDARSENSORVIEW,
  __module__ = 'osi_sensorview_pb2'
  # @@protoc_insertion_point(class_scope:osi3.LidarSensorView)
  ))
_sym_db.RegisterMessage(LidarSensorView)
_sym_db.RegisterMessage(LidarSensorView.Reflection)

CameraSensorView = _reflection.GeneratedProtocolMessageType('CameraSensorView', (_message.Message,), dict(
  DESCRIPTOR = _CAMERASENSORVIEW,
  __module__ = 'osi_sensorview_pb2'
  # @@protoc_insertion_point(class_scope:osi3.CameraSensorView)
  ))
_sym_db.RegisterMessage(CameraSensorView)

UltrasonicSensorView = _reflection.GeneratedProtocolMessageType('UltrasonicSensorView', (_message.Message,), dict(
  DESCRIPTOR = _ULTRASONICSENSORVIEW,
  __module__ = 'osi_sensorview_pb2'
  # @@protoc_insertion_point(class_scope:osi3.UltrasonicSensorView)
  ))
_sym_db.RegisterMessage(UltrasonicSensorView)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('H\001'))
# @@protoc_insertion_point(module_scope)
