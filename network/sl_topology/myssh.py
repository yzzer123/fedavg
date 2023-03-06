"""This module defines an sshd configuration."""
from distutils.spawn import find_executable
import subprocess
import os
import tempfile

from ipmininet.router.config.base import Daemon




# Generate a new ssh keypair at each run
KEYFILE = tempfile.mktemp(dir='/tmp')
PUBKEY = '%s.pub' % KEYFILE
if os.path.exists(KEYFILE):
    os.unlink(KEYFILE)
if os.path.exists(PUBKEY):
    os.unlink(PUBKEY)


class SSHd(Daemon):

    NAME = 'sshd'
    STARTUP_LINE_BASE = '{name} -D -u0'.format(name=find_executable(NAME))
    KILL_PATTERNS = (STARTUP_LINE_BASE,)

    @property
    def startup_line(self):
        return ('{base}'
                .format(base=self.STARTUP_LINE_BASE))

    @property
    def dry_run(self):
        return '%s -t' % self.startup_line

    def set_defaults(self, defaults):
        super().set_defaults(defaults)

    def build(self):
        cfg = super().build()
        cfg.authorized_keys = PUBKEY
        return cfg
