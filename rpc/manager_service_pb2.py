# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: manager_service.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import manager_message_pb2 as manager__message__pb2
import jobmanager_message_pb2 as jobmanager__message__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x15manager_service.proto\x12\x07\x66\x65\x64raft\x1a\x15manager_message.proto\x1a\x18jobmanager_message.proto2\xc0\x02\n\x0eManagerService\x12P\n\rAppendEntries\x12\x1d.fedraft.AppendEntriesRequest\x1a\x1e.fedraft.AppendEntriesResponse\"\x00\x12\x46\n\x07VoteFor\x12\x1b.fedraft.ManagerVoteRequest\x1a\x1c.fedraft.ManagerVoteResponse\"\x00\x12H\n\tJobSubmit\x12\x19.fedraft.JobSubmitRequest\x1a\x1a.fedraft.JobSubmitResponse\"\x00(\x01\x30\x01\x12J\n\x0bJobShutdown\x12\x1b.fedraft.JobShutdownRequest\x1a\x1c.fedraft.JobShutdownResponse\"\x00\x32_\n\x11JobManagerService\x12J\n\tAppendLog\x12\x1c.fedraft.AppendJobLogRequest\x1a\x1d.fedraft.AppendJobLogResponse\"\x00\x42(\n$org.bupt.fedraft.rpc.manager.serviceP\x01\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'manager_service_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n$org.bupt.fedraft.rpc.manager.serviceP\001'
  _MANAGERSERVICE._serialized_start=84
  _MANAGERSERVICE._serialized_end=404
  _JOBMANAGERSERVICE._serialized_start=406
  _JOBMANAGERSERVICE._serialized_end=501
# @@protoc_insertion_point(module_scope)
