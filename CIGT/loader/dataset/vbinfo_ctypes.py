r"""Wrapper for vb.h

Generated with:
/export/home/xiatao/miniconda3/bin/ctypesgen --cpp /home/xiatao/.conda/envs/vb/bin/x86_64-conda-linux-gnu-cc -E -Iinstall-dir/include/ -I/home/xiatao/.conda/envs/vb/include/ install-dir/include/vb/vb.h --output=1.py

Do not modify this file.
"""

__docformat__ = "restructuredtext"

# Begin preamble for Python

import ctypes
import sys
from ctypes import *  # noqa: F401, F403

_int_types = (ctypes.c_int16, ctypes.c_int32)
if hasattr(ctypes, "c_int64"):
    # Some builds of ctypes apparently do not have ctypes.c_int64
    # defined; it's a pretty good bet that these builds do not
    # have 64-bit pointers.
    _int_types += (ctypes.c_int64,)
for t in _int_types:
    if ctypes.sizeof(t) == ctypes.sizeof(ctypes.c_size_t):
        c_ptrdiff_t = t
del t
del _int_types



class UserString:
    def __init__(self, seq):
        if isinstance(seq, bytes):
            self.data = seq
        elif isinstance(seq, UserString):
            self.data = seq.data[:]
        else:
            self.data = str(seq).encode()

    def __bytes__(self):
        return self.data

    def __str__(self):
        return self.data.decode()

    def __repr__(self):
        return repr(self.data)

    def __int__(self):
        return int(self.data.decode())

    def __long__(self):
        return int(self.data.decode())

    def __float__(self):
        return float(self.data.decode())

    def __complex__(self):
        return complex(self.data.decode())

    def __hash__(self):
        return hash(self.data)

    def __le__(self, string):
        if isinstance(string, UserString):
            return self.data <= string.data
        else:
            return self.data <= string

    def __lt__(self, string):
        if isinstance(string, UserString):
            return self.data < string.data
        else:
            return self.data < string

    def __ge__(self, string):
        if isinstance(string, UserString):
            return self.data >= string.data
        else:
            return self.data >= string

    def __gt__(self, string):
        if isinstance(string, UserString):
            return self.data > string.data
        else:
            return self.data > string

    def __eq__(self, string):
        if isinstance(string, UserString):
            return self.data == string.data
        else:
            return self.data == string

    def __ne__(self, string):
        if isinstance(string, UserString):
            return self.data != string.data
        else:
            return self.data != string

    def __contains__(self, char):
        return char in self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.__class__(self.data[index])

    def __getslice__(self, start, end):
        start = max(start, 0)
        end = max(end, 0)
        return self.__class__(self.data[start:end])

    def __add__(self, other):
        if isinstance(other, UserString):
            return self.__class__(self.data + other.data)
        elif isinstance(other, bytes):
            return self.__class__(self.data + other)
        else:
            return self.__class__(self.data + str(other).encode())

    def __radd__(self, other):
        if isinstance(other, bytes):
            return self.__class__(other + self.data)
        else:
            return self.__class__(str(other).encode() + self.data)

    def __mul__(self, n):
        return self.__class__(self.data * n)

    __rmul__ = __mul__

    def __mod__(self, args):
        return self.__class__(self.data % args)

    # the following methods are defined in alphabetical order:
    def capitalize(self):
        return self.__class__(self.data.capitalize())

    def center(self, width, *args):
        return self.__class__(self.data.center(width, *args))

    def count(self, sub, start=0, end=sys.maxsize):
        return self.data.count(sub, start, end)

    def decode(self, encoding=None, errors=None):  # XXX improve this?
        if encoding:
            if errors:
                return self.__class__(self.data.decode(encoding, errors))
            else:
                return self.__class__(self.data.decode(encoding))
        else:
            return self.__class__(self.data.decode())

    def encode(self, encoding=None, errors=None):  # XXX improve this?
        if encoding:
            if errors:
                return self.__class__(self.data.encode(encoding, errors))
            else:
                return self.__class__(self.data.encode(encoding))
        else:
            return self.__class__(self.data.encode())

    def endswith(self, suffix, start=0, end=sys.maxsize):
        return self.data.endswith(suffix, start, end)

    def expandtabs(self, tabsize=8):
        return self.__class__(self.data.expandtabs(tabsize))

    def find(self, sub, start=0, end=sys.maxsize):
        return self.data.find(sub, start, end)

    def index(self, sub, start=0, end=sys.maxsize):
        return self.data.index(sub, start, end)

    def isalpha(self):
        return self.data.isalpha()

    def isalnum(self):
        return self.data.isalnum()

    def isdecimal(self):
        return self.data.isdecimal()

    def isdigit(self):
        return self.data.isdigit()

    def islower(self):
        return self.data.islower()

    def isnumeric(self):
        return self.data.isnumeric()

    def isspace(self):
        return self.data.isspace()

    def istitle(self):
        return self.data.istitle()

    def isupper(self):
        return self.data.isupper()

    def join(self, seq):
        return self.data.join(seq)

    def ljust(self, width, *args):
        return self.__class__(self.data.ljust(width, *args))

    def lower(self):
        return self.__class__(self.data.lower())

    def lstrip(self, chars=None):
        return self.__class__(self.data.lstrip(chars))

    def partition(self, sep):
        return self.data.partition(sep)

    def replace(self, old, new, maxsplit=-1):
        return self.__class__(self.data.replace(old, new, maxsplit))

    def rfind(self, sub, start=0, end=sys.maxsize):
        return self.data.rfind(sub, start, end)

    def rindex(self, sub, start=0, end=sys.maxsize):
        return self.data.rindex(sub, start, end)

    def rjust(self, width, *args):
        return self.__class__(self.data.rjust(width, *args))

    def rpartition(self, sep):
        return self.data.rpartition(sep)

    def rstrip(self, chars=None):
        return self.__class__(self.data.rstrip(chars))

    def split(self, sep=None, maxsplit=-1):
        return self.data.split(sep, maxsplit)

    def rsplit(self, sep=None, maxsplit=-1):
        return self.data.rsplit(sep, maxsplit)

    def splitlines(self, keepends=0):
        return self.data.splitlines(keepends)

    def startswith(self, prefix, start=0, end=sys.maxsize):
        return self.data.startswith(prefix, start, end)

    def strip(self, chars=None):
        return self.__class__(self.data.strip(chars))

    def swapcase(self):
        return self.__class__(self.data.swapcase())

    def title(self):
        return self.__class__(self.data.title())

    def translate(self, *args):
        return self.__class__(self.data.translate(*args))

    def upper(self):
        return self.__class__(self.data.upper())

    def zfill(self, width):
        return self.__class__(self.data.zfill(width))


class MutableString(UserString):
    """mutable string objects

    Python strings are immutable objects.  This has the advantage, that
    strings may be used as dictionary keys.  If this property isn't needed
    and you insist on changing string values in place instead, you may cheat
    and use MutableString.

    But the purpose of this class is an educational one: to prevent
    people from inventing their own mutable string class derived
    from UserString and than forget thereby to remove (override) the
    __hash__ method inherited from UserString.  This would lead to
    errors that would be very hard to track down.

    A faster and better solution is to rewrite your program using lists."""

    def __init__(self, string=""):
        self.data = string

    def __hash__(self):
        raise TypeError("unhashable type (it is mutable)")

    def __setitem__(self, index, sub):
        if index < 0:
            index += len(self.data)
        if index < 0 or index >= len(self.data):
            raise IndexError
        self.data = self.data[:index] + sub + self.data[index + 1 :]

    def __delitem__(self, index):
        if index < 0:
            index += len(self.data)
        if index < 0 or index >= len(self.data):
            raise IndexError
        self.data = self.data[:index] + self.data[index + 1 :]

    def __setslice__(self, start, end, sub):
        start = max(start, 0)
        end = max(end, 0)
        if isinstance(sub, UserString):
            self.data = self.data[:start] + sub.data + self.data[end:]
        elif isinstance(sub, bytes):
            self.data = self.data[:start] + sub + self.data[end:]
        else:
            self.data = self.data[:start] + str(sub).encode() + self.data[end:]

    def __delslice__(self, start, end):
        start = max(start, 0)
        end = max(end, 0)
        self.data = self.data[:start] + self.data[end:]

    def immutable(self):
        return UserString(self.data)

    def __iadd__(self, other):
        if isinstance(other, UserString):
            self.data += other.data
        elif isinstance(other, bytes):
            self.data += other
        else:
            self.data += str(other).encode()
        return self

    def __imul__(self, n):
        self.data *= n
        return self


class String(MutableString, ctypes.Union):

    _fields_ = [("raw", ctypes.POINTER(ctypes.c_char)), ("data", ctypes.c_char_p)]

    def __init__(self, obj=b""):
        if isinstance(obj, (bytes, UserString)):
            self.data = bytes(obj)
        else:
            self.raw = obj

    def __len__(self):
        return self.data and len(self.data) or 0

    def from_param(cls, obj):
        # Convert None or 0
        if obj is None or obj == 0:
            return cls(ctypes.POINTER(ctypes.c_char)())

        # Convert from String
        elif isinstance(obj, String):
            return obj

        # Convert from bytes
        elif isinstance(obj, bytes):
            return cls(obj)

        # Convert from str
        elif isinstance(obj, str):
            return cls(obj.encode())

        # Convert from c_char_p
        elif isinstance(obj, ctypes.c_char_p):
            return obj

        # Convert from POINTER(ctypes.c_char)
        elif isinstance(obj, ctypes.POINTER(ctypes.c_char)):
            return obj

        # Convert from raw pointer
        elif isinstance(obj, int):
            return cls(ctypes.cast(obj, ctypes.POINTER(ctypes.c_char)))

        # Convert from ctypes.c_char array
        elif isinstance(obj, ctypes.c_char * len(obj)):
            return obj

        # Convert from object
        else:
            return String.from_param(obj._as_parameter_)

    from_param = classmethod(from_param)


def ReturnString(obj, func=None, arguments=None):
    return String.from_param(obj)


# As of ctypes 1.0, ctypes does not support custom error-checking
# functions on callbacks, nor does it support custom datatypes on
# callbacks, so we must ensure that all callbacks return
# primitive datatypes.
#
# Non-primitive return values wrapped with UNCHECKED won't be
# typechecked, and will be converted to ctypes.c_void_p.
def UNCHECKED(type):
    if hasattr(type, "_type_") and isinstance(type._type_, str) and type._type_ != "P":
        return type
    else:
        return ctypes.c_void_p


# ctypes doesn't have direct support for variadic functions, so we have to write
# our own wrapper class
class _variadic_function(object):
    def __init__(self, func, restype, argtypes, errcheck):
        self.func = func
        self.func.restype = restype
        self.argtypes = argtypes
        if errcheck:
            self.func.errcheck = errcheck

    def _as_parameter_(self):
        # So we can pass this variadic function as a function pointer
        return self.func

    def __call__(self, *args):
        fixed_args = []
        i = 0
        for argtype in self.argtypes:
            # Typecheck what we can
            fixed_args.append(argtype.from_param(args[i]))
            i += 1
        return self.func(*fixed_args + list(args[i:]))


def ord_if_char(value):
    """
    Simple helper used for casts to simple builtin types:  if the argument is a
    string type, it will be converted to it's ordinal value.

    This function will raise an exception if the argument is string with more
    than one characters.
    """
    return ord(value) if (isinstance(value, bytes) or isinstance(value, str)) else value

# End preamble

_libs = {}
_libdirs = []

# Begin loader

"""
Load libraries - appropriately for all our supported platforms
"""
# ----------------------------------------------------------------------------
# Copyright (c) 2008 David James
# Copyright (c) 2006-2008 Alex Holkner
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#  * Neither the name of pyglet nor the names of its
#    contributors may be used to endorse or promote products
#    derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ----------------------------------------------------------------------------

import ctypes
import ctypes.util
import glob
import os.path
import platform
import re
import sys


def _environ_path(name):
    """Split an environment variable into a path-like list elements"""
    if name in os.environ:
        return os.environ[name].split(":")
    return []


class LibraryLoader:
    """
    A base class For loading of libraries ;-)
    Subclasses load libraries for specific platforms.
    """

    # library names formatted specifically for platforms
    name_formats = ["%s"]

    class Lookup:
        """Looking up calling conventions for a platform"""

        mode = ctypes.DEFAULT_MODE

        def __init__(self, path):
            super(LibraryLoader.Lookup, self).__init__()
            self.access = dict(cdecl=ctypes.CDLL(path, self.mode))

        def get(self, name, calling_convention="cdecl"):
            """Return the given name according to the selected calling convention"""
            if calling_convention not in self.access:
                raise LookupError(
                    "Unknown calling convention '{}' for function '{}'".format(
                        calling_convention, name
                    )
                )
            return getattr(self.access[calling_convention], name)

        def has(self, name, calling_convention="cdecl"):
            """Return True if this given calling convention finds the given 'name'"""
            if calling_convention not in self.access:
                return False
            return hasattr(self.access[calling_convention], name)

        def __getattr__(self, name):
            return getattr(self.access["cdecl"], name)

    def __init__(self):
        self.other_dirs = []

    def __call__(self, libname):
        """Given the name of a library, load it."""
        paths = self.getpaths(libname)

        for path in paths:
            # noinspection PyBroadException
            try:
                return self.Lookup(path)
            except Exception:  # pylint: disable=broad-except
                pass

        raise ImportError("Could not load %s." % libname)

    def getpaths(self, libname):
        """Return a list of paths where the library might be found."""
        if os.path.isabs(libname):
            yield libname
        else:
            # search through a prioritized series of locations for the library

            # we first search any specific directories identified by user
            for dir_i in self.other_dirs:
                for fmt in self.name_formats:
                    # dir_i should be absolute already
                    yield os.path.join(dir_i, fmt % libname)

            # check if this code is even stored in a physical file
            try:
                this_file = __file__
            except NameError:
                this_file = None

            # then we search the directory where the generated python interface is stored
            if this_file is not None:
                for fmt in self.name_formats:
                    yield os.path.abspath(os.path.join(os.path.dirname(__file__), fmt % libname))

            # now, use the ctypes tools to try to find the library
            for fmt in self.name_formats:
                path = ctypes.util.find_library(fmt % libname)
                if path:
                    yield path

            # then we search all paths identified as platform-specific lib paths
            for path in self.getplatformpaths(libname):
                yield path

            # Finally, we'll try the users current working directory
            for fmt in self.name_formats:
                yield os.path.abspath(os.path.join(os.path.curdir, fmt % libname))

    def getplatformpaths(self, _libname):  # pylint: disable=no-self-use
        """Return all the library paths available in this platform"""
        return []


# Darwin (Mac OS X)


class DarwinLibraryLoader(LibraryLoader):
    """Library loader for MacOS"""

    name_formats = [
        "lib%s.dylib",
        "lib%s.so",
        "lib%s.bundle",
        "%s.dylib",
        "%s.so",
        "%s.bundle",
        "%s",
    ]

    class Lookup(LibraryLoader.Lookup):
        """
        Looking up library files for this platform (Darwin aka MacOS)
        """

        # Darwin requires dlopen to be called with mode RTLD_GLOBAL instead
        # of the default RTLD_LOCAL.  Without this, you end up with
        # libraries not being loadable, resulting in "Symbol not found"
        # errors
        mode = ctypes.RTLD_GLOBAL

    def getplatformpaths(self, libname):
        if os.path.pathsep in libname:
            names = [libname]
        else:
            names = [fmt % libname for fmt in self.name_formats]

        for directory in self.getdirs(libname):
            for name in names:
                yield os.path.join(directory, name)

    @staticmethod
    def getdirs(libname):
        """Implements the dylib search as specified in Apple documentation:

        http://developer.apple.com/documentation/DeveloperTools/Conceptual/
            DynamicLibraries/Articles/DynamicLibraryUsageGuidelines.html

        Before commencing the standard search, the method first checks
        the bundle's ``Frameworks`` directory if the application is running
        within a bundle (OS X .app).
        """

        dyld_fallback_library_path = _environ_path("DYLD_FALLBACK_LIBRARY_PATH")
        if not dyld_fallback_library_path:
            dyld_fallback_library_path = [
                os.path.expanduser("~/lib"),
                "/usr/local/lib",
                "/usr/lib",
            ]

        dirs = []

        if "/" in libname:
            dirs.extend(_environ_path("DYLD_LIBRARY_PATH"))
        else:
            dirs.extend(_environ_path("LD_LIBRARY_PATH"))
            dirs.extend(_environ_path("DYLD_LIBRARY_PATH"))
            dirs.extend(_environ_path("LD_RUN_PATH"))

        if hasattr(sys, "frozen") and getattr(sys, "frozen") == "macosx_app":
            dirs.append(os.path.join(os.environ["RESOURCEPATH"], "..", "Frameworks"))

        dirs.extend(dyld_fallback_library_path)

        return dirs


# Posix


class PosixLibraryLoader(LibraryLoader):
    """Library loader for POSIX-like systems (including Linux)"""

    _ld_so_cache = None

    _include = re.compile(r"^\s*include\s+(?P<pattern>.*)")

    name_formats = ["lib%s.so", "%s.so", "%s"]

    class _Directories(dict):
        """Deal with directories"""

        def __init__(self):
            dict.__init__(self)
            self.order = 0

        def add(self, directory):
            """Add a directory to our current set of directories"""
            if len(directory) > 1:
                directory = directory.rstrip(os.path.sep)
            # only adds and updates order if exists and not already in set
            if not os.path.exists(directory):
                return
            order = self.setdefault(directory, self.order)
            if order == self.order:
                self.order += 1

        def extend(self, directories):
            """Add a list of directories to our set"""
            for a_dir in directories:
                self.add(a_dir)

        def ordered(self):
            """Sort the list of directories"""
            return (i[0] for i in sorted(self.items(), key=lambda d: d[1]))

    def _get_ld_so_conf_dirs(self, conf, dirs):
        """
        Recursive function to help parse all ld.so.conf files, including proper
        handling of the `include` directive.
        """

        try:
            with open(conf) as fileobj:
                for dirname in fileobj:
                    dirname = dirname.strip()
                    if not dirname:
                        continue

                    match = self._include.match(dirname)
                    if not match:
                        dirs.add(dirname)
                    else:
                        for dir2 in glob.glob(match.group("pattern")):
                            self._get_ld_so_conf_dirs(dir2, dirs)
        except IOError:
            pass

    def _create_ld_so_cache(self):
        # Recreate search path followed by ld.so.  This is going to be
        # slow to build, and incorrect (ld.so uses ld.so.cache, which may
        # not be up-to-date).  Used only as fallback for distros without
        # /sbin/ldconfig.
        #
        # We assume the DT_RPATH and DT_RUNPATH binary sections are omitted.

        directories = self._Directories()
        for name in (
            "LD_LIBRARY_PATH",
            "SHLIB_PATH",  # HP-UX
            "LIBPATH",  # OS/2, AIX
            "LIBRARY_PATH",  # BE/OS
        ):
            if name in os.environ:
                directories.extend(os.environ[name].split(os.pathsep))

        self._get_ld_so_conf_dirs("/etc/ld.so.conf", directories)

        bitage = platform.architecture()[0]

        unix_lib_dirs_list = []
        if bitage.startswith("64"):
            # prefer 64 bit if that is our arch
            unix_lib_dirs_list += ["/lib64", "/usr/lib64"]

        # must include standard libs, since those paths are also used by 64 bit
        # installs
        unix_lib_dirs_list += ["/lib", "/usr/lib"]
        if sys.platform.startswith("linux"):
            # Try and support multiarch work in Ubuntu
            # https://wiki.ubuntu.com/MultiarchSpec
            if bitage.startswith("32"):
                # Assume Intel/AMD x86 compat
                unix_lib_dirs_list += ["/lib/i386-linux-gnu", "/usr/lib/i386-linux-gnu"]
            elif bitage.startswith("64"):
                # Assume Intel/AMD x86 compatible
                unix_lib_dirs_list += [
                    "/lib/x86_64-linux-gnu",
                    "/usr/lib/x86_64-linux-gnu",
                ]
            else:
                # guess...
                unix_lib_dirs_list += glob.glob("/lib/*linux-gnu")
        directories.extend(unix_lib_dirs_list)

        cache = {}
        lib_re = re.compile(r"lib(.*)\.s[ol]")
        # ext_re = re.compile(r"\.s[ol]$")
        for our_dir in directories.ordered():
            try:
                for path in glob.glob("%s/*.s[ol]*" % our_dir):
                    file = os.path.basename(path)

                    # Index by filename
                    cache_i = cache.setdefault(file, set())
                    cache_i.add(path)

                    # Index by library name
                    match = lib_re.match(file)
                    if match:
                        library = match.group(1)
                        cache_i = cache.setdefault(library, set())
                        cache_i.add(path)
            except OSError:
                pass

        self._ld_so_cache = cache

    def getplatformpaths(self, libname):
        if self._ld_so_cache is None:
            self._create_ld_so_cache()

        result = self._ld_so_cache.get(libname, set())
        for i in result:
            # we iterate through all found paths for library, since we may have
            # actually found multiple architectures or other library types that
            # may not load
            yield i


# Windows


class WindowsLibraryLoader(LibraryLoader):
    """Library loader for Microsoft Windows"""

    name_formats = ["%s.dll", "lib%s.dll", "%slib.dll", "%s"]

    class Lookup(LibraryLoader.Lookup):
        """Lookup class for Windows libraries..."""

        def __init__(self, path):
            super(WindowsLibraryLoader.Lookup, self).__init__(path)
            self.access["stdcall"] = ctypes.windll.LoadLibrary(path)


# Platform switching

# If your value of sys.platform does not appear in this dict, please contact
# the Ctypesgen maintainers.

loaderclass = {
    "darwin": DarwinLibraryLoader,
    "cygwin": WindowsLibraryLoader,
    "win32": WindowsLibraryLoader,
    "msys": WindowsLibraryLoader,
}

load_library = loaderclass.get(sys.platform, PosixLibraryLoader)()


def add_library_search_dirs(other_dirs):
    """
    Add libraries to search paths.
    If library paths are relative, convert them to absolute with respect to this
    file's directory
    """
    for path in other_dirs:
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        load_library.other_dirs.append(path)


del loaderclass

# End loader

add_library_search_dirs([])

# No libraries

# No modules

# install-dir/include/mol/data_struct.h: 39
class struct_tree_node_struct(Structure):
    pass

struct_tree_node_struct.__slots__ = [
    'data',
    'key',
    'left_leaf',
    'right_leaf',
]
struct_tree_node_struct._fields_ = [
    ('data', c_int),
    ('key', c_int),
    ('left_leaf', POINTER(struct_tree_node_struct)),
    ('right_leaf', POINTER(struct_tree_node_struct)),
]

tree_node = struct_tree_node_struct# install-dir/include/mol/data_struct.h: 45

# install-dir/include/mol/data_struct.h: 52
class struct_anon_31(Structure):
    pass

struct_anon_31.__slots__ = [
    'node_num',
    'node_list',
    'head',
]
struct_anon_31._fields_ = [
    ('node_num', c_int),
    ('node_list', POINTER(tree_node)),
    ('head', POINTER(tree_node)),
]

qtree = struct_anon_31# install-dir/include/mol/data_struct.h: 52

# /home/xiatao/.conda/envs/vb/include/xc.h: 312
class struct_xc_func_type(Structure):
    pass

# /home/xiatao/.conda/envs/vb/include/xc.h: 160
class struct_anon_40(Structure):
    pass

struct_anon_40.__slots__ = [
    'ref',
    'doi',
    'bibtex',
    'key',
]
struct_anon_40._fields_ = [
    ('ref', String),
    ('doi', String),
    ('bibtex', String),
    ('key', String),
]

func_reference_type = struct_anon_40# /home/xiatao/.conda/envs/vb/include/xc.h: 160

# /home/xiatao/.conda/envs/vb/include/xc.h: 177
class struct_anon_41(Structure):
    pass

struct_anon_41.__slots__ = [
    'n',
    'names',
    'descriptions',
    'values',
    'set',
]
struct_anon_41._fields_ = [
    ('n', c_int),
    ('names', POINTER(POINTER(c_char))),
    ('descriptions', POINTER(POINTER(c_char))),
    ('values', POINTER(c_double)),
    ('set', CFUNCTYPE(UNCHECKED(None), POINTER(struct_xc_func_type), POINTER(c_double))),
]

func_params_type = struct_anon_41# /home/xiatao/.conda/envs/vb/include/xc.h: 177

# /home/xiatao/.conda/envs/vb/include/xc.h: 192
class struct_anon_42(Structure):
    pass

struct_anon_42.__slots__ = [
    'zk',
    'vrho',
    'v2rho2',
    'v3rho3',
    'v4rho4',
]
struct_anon_42._fields_ = [
    ('zk', POINTER(c_double)),
    ('vrho', POINTER(c_double)),
    ('v2rho2', POINTER(c_double)),
    ('v3rho3', POINTER(c_double)),
    ('v4rho4', POINTER(c_double)),
]

xc_lda_out_params = struct_anon_42# /home/xiatao/.conda/envs/vb/include/xc.h: 192

# /home/xiatao/.conda/envs/vb/include/xc.h: 205
class struct_anon_43(Structure):
    pass

struct_anon_43.__slots__ = [
    'zk',
    'vrho',
    'vsigma',
    'v2rho2',
    'v2rhosigma',
    'v2sigma2',
    'v3rho3',
    'v3rho2sigma',
    'v3rhosigma2',
    'v3sigma3',
    'v4rho4',
    'v4rho3sigma',
    'v4rho2sigma2',
    'v4rhosigma3',
    'v4sigma4',
]
struct_anon_43._fields_ = [
    ('zk', POINTER(c_double)),
    ('vrho', POINTER(c_double)),
    ('vsigma', POINTER(c_double)),
    ('v2rho2', POINTER(c_double)),
    ('v2rhosigma', POINTER(c_double)),
    ('v2sigma2', POINTER(c_double)),
    ('v3rho3', POINTER(c_double)),
    ('v3rho2sigma', POINTER(c_double)),
    ('v3rhosigma2', POINTER(c_double)),
    ('v3sigma3', POINTER(c_double)),
    ('v4rho4', POINTER(c_double)),
    ('v4rho3sigma', POINTER(c_double)),
    ('v4rho2sigma2', POINTER(c_double)),
    ('v4rhosigma3', POINTER(c_double)),
    ('v4sigma4', POINTER(c_double)),
]

xc_gga_out_params = struct_anon_43# /home/xiatao/.conda/envs/vb/include/xc.h: 205

# /home/xiatao/.conda/envs/vb/include/xc.h: 231
class struct_anon_44(Structure):
    pass

struct_anon_44.__slots__ = [
    'zk',
    'vrho',
    'vsigma',
    'vlapl',
    'vtau',
    'v2rho2',
    'v2rhosigma',
    'v2rholapl',
    'v2rhotau',
    'v2sigma2',
    'v2sigmalapl',
    'v2sigmatau',
    'v2lapl2',
    'v2lapltau',
    'v2tau2',
    'v3rho3',
    'v3rho2sigma',
    'v3rho2lapl',
    'v3rho2tau',
    'v3rhosigma2',
    'v3rhosigmalapl',
    'v3rhosigmatau',
    'v3rholapl2',
    'v3rholapltau',
    'v3rhotau2',
    'v3sigma3',
    'v3sigma2lapl',
    'v3sigma2tau',
    'v3sigmalapl2',
    'v3sigmalapltau',
    'v3sigmatau2',
    'v3lapl3',
    'v3lapl2tau',
    'v3lapltau2',
    'v3tau3',
    'v4rho4',
    'v4rho3sigma',
    'v4rho3lapl',
    'v4rho3tau',
    'v4rho2sigma2',
    'v4rho2sigmalapl',
    'v4rho2sigmatau',
    'v4rho2lapl2',
    'v4rho2lapltau',
    'v4rho2tau2',
    'v4rhosigma3',
    'v4rhosigma2lapl',
    'v4rhosigma2tau',
    'v4rhosigmalapl2',
    'v4rhosigmalapltau',
    'v4rhosigmatau2',
    'v4rholapl3',
    'v4rholapl2tau',
    'v4rholapltau2',
    'v4rhotau3',
    'v4sigma4',
    'v4sigma3lapl',
    'v4sigma3tau',
    'v4sigma2lapl2',
    'v4sigma2lapltau',
    'v4sigma2tau2',
    'v4sigmalapl3',
    'v4sigmalapl2tau',
    'v4sigmalapltau2',
    'v4sigmatau3',
    'v4lapl4',
    'v4lapl3tau',
    'v4lapl2tau2',
    'v4lapltau3',
    'v4tau4',
]
struct_anon_44._fields_ = [
    ('zk', POINTER(c_double)),
    ('vrho', POINTER(c_double)),
    ('vsigma', POINTER(c_double)),
    ('vlapl', POINTER(c_double)),
    ('vtau', POINTER(c_double)),
    ('v2rho2', POINTER(c_double)),
    ('v2rhosigma', POINTER(c_double)),
    ('v2rholapl', POINTER(c_double)),
    ('v2rhotau', POINTER(c_double)),
    ('v2sigma2', POINTER(c_double)),
    ('v2sigmalapl', POINTER(c_double)),
    ('v2sigmatau', POINTER(c_double)),
    ('v2lapl2', POINTER(c_double)),
    ('v2lapltau', POINTER(c_double)),
    ('v2tau2', POINTER(c_double)),
    ('v3rho3', POINTER(c_double)),
    ('v3rho2sigma', POINTER(c_double)),
    ('v3rho2lapl', POINTER(c_double)),
    ('v3rho2tau', POINTER(c_double)),
    ('v3rhosigma2', POINTER(c_double)),
    ('v3rhosigmalapl', POINTER(c_double)),
    ('v3rhosigmatau', POINTER(c_double)),
    ('v3rholapl2', POINTER(c_double)),
    ('v3rholapltau', POINTER(c_double)),
    ('v3rhotau2', POINTER(c_double)),
    ('v3sigma3', POINTER(c_double)),
    ('v3sigma2lapl', POINTER(c_double)),
    ('v3sigma2tau', POINTER(c_double)),
    ('v3sigmalapl2', POINTER(c_double)),
    ('v3sigmalapltau', POINTER(c_double)),
    ('v3sigmatau2', POINTER(c_double)),
    ('v3lapl3', POINTER(c_double)),
    ('v3lapl2tau', POINTER(c_double)),
    ('v3lapltau2', POINTER(c_double)),
    ('v3tau3', POINTER(c_double)),
    ('v4rho4', POINTER(c_double)),
    ('v4rho3sigma', POINTER(c_double)),
    ('v4rho3lapl', POINTER(c_double)),
    ('v4rho3tau', POINTER(c_double)),
    ('v4rho2sigma2', POINTER(c_double)),
    ('v4rho2sigmalapl', POINTER(c_double)),
    ('v4rho2sigmatau', POINTER(c_double)),
    ('v4rho2lapl2', POINTER(c_double)),
    ('v4rho2lapltau', POINTER(c_double)),
    ('v4rho2tau2', POINTER(c_double)),
    ('v4rhosigma3', POINTER(c_double)),
    ('v4rhosigma2lapl', POINTER(c_double)),
    ('v4rhosigma2tau', POINTER(c_double)),
    ('v4rhosigmalapl2', POINTER(c_double)),
    ('v4rhosigmalapltau', POINTER(c_double)),
    ('v4rhosigmatau2', POINTER(c_double)),
    ('v4rholapl3', POINTER(c_double)),
    ('v4rholapl2tau', POINTER(c_double)),
    ('v4rholapltau2', POINTER(c_double)),
    ('v4rhotau3', POINTER(c_double)),
    ('v4sigma4', POINTER(c_double)),
    ('v4sigma3lapl', POINTER(c_double)),
    ('v4sigma3tau', POINTER(c_double)),
    ('v4sigma2lapl2', POINTER(c_double)),
    ('v4sigma2lapltau', POINTER(c_double)),
    ('v4sigma2tau2', POINTER(c_double)),
    ('v4sigmalapl3', POINTER(c_double)),
    ('v4sigmalapl2tau', POINTER(c_double)),
    ('v4sigmalapltau2', POINTER(c_double)),
    ('v4sigmatau3', POINTER(c_double)),
    ('v4lapl4', POINTER(c_double)),
    ('v4lapl3tau', POINTER(c_double)),
    ('v4lapl2tau2', POINTER(c_double)),
    ('v4lapltau3', POINTER(c_double)),
    ('v4tau4', POINTER(c_double)),
]

xc_mgga_out_params = struct_anon_44# /home/xiatao/.conda/envs/vb/include/xc.h: 231

xc_lda_funcs = CFUNCTYPE(UNCHECKED(None), POINTER(struct_xc_func_type), c_size_t, POINTER(c_double), POINTER(xc_lda_out_params))# /home/xiatao/.conda/envs/vb/include/xc.h: 234

# /home/xiatao/.conda/envs/vb/include/xc.h: 241
class struct_anon_45(Structure):
    pass

struct_anon_45.__slots__ = [
    'unpol',
    'pol',
]
struct_anon_45._fields_ = [
    ('unpol', xc_lda_funcs * int(5)),
    ('pol', xc_lda_funcs * int(5)),
]

xc_lda_funcs_variants = struct_anon_45# /home/xiatao/.conda/envs/vb/include/xc.h: 241

xc_gga_funcs = CFUNCTYPE(UNCHECKED(None), POINTER(struct_xc_func_type), c_size_t, POINTER(c_double), POINTER(c_double), POINTER(xc_gga_out_params))# /home/xiatao/.conda/envs/vb/include/xc.h: 244

# /home/xiatao/.conda/envs/vb/include/xc.h: 251
class struct_anon_46(Structure):
    pass

struct_anon_46.__slots__ = [
    'unpol',
    'pol',
]
struct_anon_46._fields_ = [
    ('unpol', xc_gga_funcs * int(5)),
    ('pol', xc_gga_funcs * int(5)),
]

xc_gga_funcs_variants = struct_anon_46# /home/xiatao/.conda/envs/vb/include/xc.h: 251

xc_mgga_funcs = CFUNCTYPE(UNCHECKED(None), POINTER(struct_xc_func_type), c_size_t, POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(xc_mgga_out_params))# /home/xiatao/.conda/envs/vb/include/xc.h: 254

# /home/xiatao/.conda/envs/vb/include/xc.h: 260
class struct_anon_47(Structure):
    pass

struct_anon_47.__slots__ = [
    'unpol',
    'pol',
]
struct_anon_47._fields_ = [
    ('unpol', xc_mgga_funcs * int(5)),
    ('pol', xc_mgga_funcs * int(5)),
]

xc_mgga_funcs_variants = struct_anon_47# /home/xiatao/.conda/envs/vb/include/xc.h: 260

# /home/xiatao/.conda/envs/vb/include/xc.h: 283
class struct_anon_48(Structure):
    pass

struct_anon_48.__slots__ = [
    'number',
    'kind',
    'name',
    'family',
    'refs',
    'flags',
    'dens_threshold',
    'ext_params',
    'init',
    'end',
    'lda',
    'gga',
    'mgga',
]
struct_anon_48._fields_ = [
    ('number', c_int),
    ('kind', c_int),
    ('name', String),
    ('family', c_int),
    ('refs', POINTER(func_reference_type) * int(5)),
    ('flags', c_int),
    ('dens_threshold', c_double),
    ('ext_params', func_params_type),
    ('init', CFUNCTYPE(UNCHECKED(None), POINTER(struct_xc_func_type))),
    ('end', CFUNCTYPE(UNCHECKED(None), POINTER(struct_xc_func_type))),
    ('lda', POINTER(xc_lda_funcs_variants)),
    ('gga', POINTER(xc_gga_funcs_variants)),
    ('mgga', POINTER(xc_mgga_funcs_variants)),
]

xc_func_info_type = struct_anon_48# /home/xiatao/.conda/envs/vb/include/xc.h: 283

# /home/xiatao/.conda/envs/vb/include/xc.h: 304
class struct_xc_dimensions(Structure):
    pass

struct_xc_dimensions.__slots__ = [
    'rho',
    'sigma',
    'lapl',
    'tau',
    'zk',
    'vrho',
    'vsigma',
    'vlapl',
    'vtau',
    'v2rho2',
    'v2rhosigma',
    'v2rholapl',
    'v2rhotau',
    'v2sigma2',
    'v2sigmalapl',
    'v2sigmatau',
    'v2lapl2',
    'v2lapltau',
    'v2tau2',
    'v3rho3',
    'v3rho2sigma',
    'v3rho2lapl',
    'v3rho2tau',
    'v3rhosigma2',
    'v3rhosigmalapl',
    'v3rhosigmatau',
    'v3rholapl2',
    'v3rholapltau',
    'v3rhotau2',
    'v3sigma3',
    'v3sigma2lapl',
    'v3sigma2tau',
    'v3sigmalapl2',
    'v3sigmalapltau',
    'v3sigmatau2',
    'v3lapl3',
    'v3lapl2tau',
    'v3lapltau2',
    'v3tau3',
    'v4rho4',
    'v4rho3sigma',
    'v4rho3lapl',
    'v4rho3tau',
    'v4rho2sigma2',
    'v4rho2sigmalapl',
    'v4rho2sigmatau',
    'v4rho2lapl2',
    'v4rho2lapltau',
    'v4rho2tau2',
    'v4rhosigma3',
    'v4rhosigma2lapl',
    'v4rhosigma2tau',
    'v4rhosigmalapl2',
    'v4rhosigmalapltau',
    'v4rhosigmatau2',
    'v4rholapl3',
    'v4rholapl2tau',
    'v4rholapltau2',
    'v4rhotau3',
    'v4sigma4',
    'v4sigma3lapl',
    'v4sigma3tau',
    'v4sigma2lapl2',
    'v4sigma2lapltau',
    'v4sigma2tau2',
    'v4sigmalapl3',
    'v4sigmalapl2tau',
    'v4sigmalapltau2',
    'v4sigmatau3',
    'v4lapl4',
    'v4lapl3tau',
    'v4lapl2tau2',
    'v4lapltau3',
    'v4tau4',
]
struct_xc_dimensions._fields_ = [
    ('rho', c_int),
    ('sigma', c_int),
    ('lapl', c_int),
    ('tau', c_int),
    ('zk', c_int),
    ('vrho', c_int),
    ('vsigma', c_int),
    ('vlapl', c_int),
    ('vtau', c_int),
    ('v2rho2', c_int),
    ('v2rhosigma', c_int),
    ('v2rholapl', c_int),
    ('v2rhotau', c_int),
    ('v2sigma2', c_int),
    ('v2sigmalapl', c_int),
    ('v2sigmatau', c_int),
    ('v2lapl2', c_int),
    ('v2lapltau', c_int),
    ('v2tau2', c_int),
    ('v3rho3', c_int),
    ('v3rho2sigma', c_int),
    ('v3rho2lapl', c_int),
    ('v3rho2tau', c_int),
    ('v3rhosigma2', c_int),
    ('v3rhosigmalapl', c_int),
    ('v3rhosigmatau', c_int),
    ('v3rholapl2', c_int),
    ('v3rholapltau', c_int),
    ('v3rhotau2', c_int),
    ('v3sigma3', c_int),
    ('v3sigma2lapl', c_int),
    ('v3sigma2tau', c_int),
    ('v3sigmalapl2', c_int),
    ('v3sigmalapltau', c_int),
    ('v3sigmatau2', c_int),
    ('v3lapl3', c_int),
    ('v3lapl2tau', c_int),
    ('v3lapltau2', c_int),
    ('v3tau3', c_int),
    ('v4rho4', c_int),
    ('v4rho3sigma', c_int),
    ('v4rho3lapl', c_int),
    ('v4rho3tau', c_int),
    ('v4rho2sigma2', c_int),
    ('v4rho2sigmalapl', c_int),
    ('v4rho2sigmatau', c_int),
    ('v4rho2lapl2', c_int),
    ('v4rho2lapltau', c_int),
    ('v4rho2tau2', c_int),
    ('v4rhosigma3', c_int),
    ('v4rhosigma2lapl', c_int),
    ('v4rhosigma2tau', c_int),
    ('v4rhosigmalapl2', c_int),
    ('v4rhosigmalapltau', c_int),
    ('v4rhosigmatau2', c_int),
    ('v4rholapl3', c_int),
    ('v4rholapl2tau', c_int),
    ('v4rholapltau2', c_int),
    ('v4rhotau3', c_int),
    ('v4sigma4', c_int),
    ('v4sigma3lapl', c_int),
    ('v4sigma3tau', c_int),
    ('v4sigma2lapl2', c_int),
    ('v4sigma2lapltau', c_int),
    ('v4sigma2tau2', c_int),
    ('v4sigmalapl3', c_int),
    ('v4sigmalapl2tau', c_int),
    ('v4sigmalapltau2', c_int),
    ('v4sigmatau3', c_int),
    ('v4lapl4', c_int),
    ('v4lapl3tau', c_int),
    ('v4lapl2tau2', c_int),
    ('v4lapltau3', c_int),
    ('v4tau4', c_int),
]

xc_dimensions = struct_xc_dimensions# /home/xiatao/.conda/envs/vb/include/xc.h: 309

struct_xc_func_type.__slots__ = [
    'info',
    'nspin',
    'n_func_aux',
    'func_aux',
    'mix_coef',
    'cam_omega',
    'cam_alpha',
    'cam_beta',
    'nlc_b',
    'nlc_C',
    'dim',
    'params',
    'dens_threshold',
    'zeta_threshold',
    'sigma_threshold',
    'tau_threshold',
]
struct_xc_func_type._fields_ = [
    ('info', POINTER(xc_func_info_type)),
    ('nspin', c_int),
    ('n_func_aux', c_int),
    ('func_aux', POINTER(POINTER(struct_xc_func_type))),
    ('mix_coef', POINTER(c_double)),
    ('cam_omega', c_double),
    ('cam_alpha', c_double),
    ('cam_beta', c_double),
    ('nlc_b', c_double),
    ('nlc_C', c_double),
    ('dim', xc_dimensions),
    ('params', POINTER(None)),
    ('dens_threshold', c_double),
    ('zeta_threshold', c_double),
    ('sigma_threshold', c_double),
    ('tau_threshold', c_double),
]

xc_func_type = struct_xc_func_type# /home/xiatao/.conda/envs/vb/include/xc.h: 348

List = POINTER(c_int)# install-dir/include/mol/mol.h: 66

Vector = POINTER(c_double)# install-dir/include/mol/mol.h: 67

Matrix = POINTER(POINTER(c_double))# install-dir/include/mol/mol.h: 68

Tensor4D = POINTER(POINTER(POINTER(POINTER(c_double))))# install-dir/include/mol/mol.h: 70

# install-dir/include/mol/mol.h: 167
class struct_ShellPair(Structure):
    pass

shell_pair = POINTER(struct_ShellPair)# install-dir/include/mol/mol.h: 71

# install-dir/include/mol/mol.h: 181
class struct_PairInfo(Structure):
    pass

pair_info = POINTER(struct_PairInfo)# install-dir/include/mol/mol.h: 72

# install-dir/include/mol/mol.h: 93
class struct_AtmInfo(Structure):
    pass

atm_info = POINTER(struct_AtmInfo)# install-dir/include/mol/mol.h: 73

# install-dir/include/mol/mol.h: 119
class struct_BasInfo(Structure):
    pass

bas_info = POINTER(struct_BasInfo)# install-dir/include/mol/mol.h: 74

# install-dir/include/mol/mol.h: 149
class struct_PrmInfo(Structure):
    pass

prm_info = POINTER(struct_PrmInfo)# install-dir/include/mol/mol.h: 75

# install-dir/include/mol/mol.h: 212
class struct_MolInfo(Structure):
    pass

mol_info = POINTER(struct_MolInfo)# install-dir/include/mol/mol.h: 76

struct_AtmInfo.__slots__ = [
    'natm',
    'charge',
    'mult',
    'alpha_num',
    'beta_num',
    'atm',
    'value',
    'mol_fname',
]
struct_AtmInfo._fields_ = [
    ('natm', c_int),
    ('charge', c_int),
    ('mult', c_int),
    ('alpha_num', c_int),
    ('beta_num', c_int),
    ('atm', List),
    ('value', Vector),
    ('mol_fname', String),
]

struct_BasInfo.__slots__ = [
    'atm',
    'nbas',
    'msize',
    'bas',
    'shls_p',
    'value',
    'bas_fname',
    'qoff',
]
struct_BasInfo._fields_ = [
    ('atm', atm_info),
    ('nbas', c_int),
    ('msize', c_int),
    ('bas', List),
    ('shls_p', List),
    ('value', Vector),
    ('bas_fname', String),
    ('qoff', c_int),
]

struct_PrmInfo.__slots__ = [
    'bas',
    'atm',
    'nprm',
    'msize',
    'prm',
    'shls_p',
    'value',
    'c2p_len',
    'c2p_index',
    'c2p_coe',
]
struct_PrmInfo._fields_ = [
    ('bas', bas_info),
    ('atm', atm_info),
    ('nprm', c_int),
    ('msize', c_int),
    ('prm', List),
    ('shls_p', List),
    ('value', Vector),
    ('c2p_len', c_int),
    ('c2p_index', List),
    ('c2p_coe', Vector),
]

struct_ShellPair.__slots__ = [
    'off_vec',
    'shli',
    'shlj',
    'di',
    'dj',
    'dij',
    'li',
    'lj',
    'AB',
    'xi',
    'half_xi_',
    'P',
    'PA',
    'PB',
    'K_ab',
    'extent',
    'EST',
]
struct_ShellPair._fields_ = [
    ('off_vec', c_int * int(3)),
    ('shli', c_int),
    ('shlj', c_int),
    ('di', c_int),
    ('dj', c_int),
    ('dij', c_int),
    ('li', c_int),
    ('lj', c_int),
    ('AB', c_double * int(3)),
    ('xi', POINTER(c_double)),
    ('half_xi_', POINTER(c_double)),
    ('P', POINTER(c_double)),
    ('PA', POINTER(c_double)),
    ('PB', POINTER(c_double)),
    ('K_ab', POINTER(c_double)),
    ('extent', c_double),
    ('EST', c_double),
]

struct_PairInfo.__slots__ = [
    'nbas',
    'len',
    'shp',
    'search_tree',
]
struct_PairInfo._fields_ = [
    ('nbas', c_int),
    ('len', c_int),
    ('shp', shell_pair),
    ('search_tree', POINTER(qtree)),
]

struct_MolInfo.__slots__ = [
    'atm',
    'bas',
    'prm',
    'pair',
    'prm_pair',
]
struct_MolInfo._fields_ = [
    ('atm', atm_info),
    ('bas', bas_info),
    ('prm', prm_info),
    ('pair', pair_info),
    ('prm_pair', pair_info),
]

enum_anon_49 = c_int# install-dir/include/mol/xgrids.h: 17

GRIDS_TYPE = enum_anon_49# install-dir/include/mol/xgrids.h: 17

# install-dir/include/mol/xgrids.h: 22
class struct_grid_t(Structure):
    pass

struct_grid_t.__slots__ = [
    'x',
    'y',
    'z',
    'w',
    'ia',
]
struct_grid_t._fields_ = [
    ('x', c_double),
    ('y', c_double),
    ('z', c_double),
    ('w', c_double),
    ('ia', c_int),
]

grid_t = struct_grid_t# install-dir/include/mol/xgrids.h: 22

# install-dir/include/mol/xgrids.h: 27
class struct_GridBlock(Structure):
    pass

struct_GridBlock.__slots__ = [
    'grid_num',
    'bas_num',
    'shell_num',
    'grid_list',
    'shell_list',
    'local2global',
    'global2local',
    'local2global_aux',
    'global2local_aux',
    'psi',
    'psix',
    'psiy',
    'psiz',
    'psixx',
    'psiyy',
    'psizz',
    'psixy',
    'psixz',
    'psiyz',
]
struct_GridBlock._fields_ = [
    ('grid_num', c_int),
    ('bas_num', c_int),
    ('shell_num', c_int),
    ('grid_list', List),
    ('shell_list', List),
    ('local2global', List),
    ('global2local', List),
    ('local2global_aux', List),
    ('global2local_aux', List),
    ('psi', Matrix),
    ('psix', Matrix),
    ('psiy', Matrix),
    ('psiz', Matrix),
    ('psixx', Matrix),
    ('psiyy', Matrix),
    ('psizz', Matrix),
    ('psixy', Matrix),
    ('psixz', Matrix),
    ('psiyz', Matrix),
]

grid_block = POINTER(struct_GridBlock)# install-dir/include/mol/xgrids.h: 39

# install-dir/include/mol/xgrids.h: 40
class struct_GridsInfo(Structure):
    pass

struct_GridsInfo.__slots__ = [
    'maxR',
    'grid_num',
    'grids',
    'block_num',
    'blocks',
    'bas_extent',
    'mol',
]
struct_GridsInfo._fields_ = [
    ('maxR', c_double),
    ('grid_num', c_int),
    ('grids', POINTER(grid_t)),
    ('block_num', c_int),
    ('blocks', grid_block),
    ('bas_extent', Vector),
    ('mol', mol_info),
]

grids_info = POINTER(struct_GridsInfo)# install-dir/include/mol/xgrids.h: 51

enum_Jbuilder_type = c_int# install-dir/include/fock/fock.h: 11

Jbuilder_type = enum_Jbuilder_type# install-dir/include/fock/fock.h: 11

enum_Kbuilder_type = c_int# install-dir/include/fock/fock.h: 20

Kbuilder_type = enum_Kbuilder_type# install-dir/include/fock/fock.h: 20

# install-dir/include/fock/fock.h: 21
class struct_CoulombInfo(Structure):
    pass

struct_CoulombInfo.__slots__ = [
    'type',
    'if_aux',
    'if_grids',
    'if_final',
    'batch_size',
    'J',
    'mol',
    'aux',
    'aux2e_matrix',
    'o_matrix',
    'grids',
    'o_matrix_final',
    'grids_final',
    'df_coe',
]
struct_CoulombInfo._fields_ = [
    ('type', Jbuilder_type),
    ('if_aux', c_bool),
    ('if_grids', c_bool),
    ('if_final', c_bool),
    ('batch_size', c_int),
    ('J', POINTER(Matrix)),
    ('mol', mol_info),
    ('aux', bas_info),
    ('aux2e_matrix', Matrix),
    ('o_matrix', Matrix),
    ('grids', grids_info),
    ('o_matrix_final', Matrix),
    ('grids_final', grids_info),
    ('df_coe', POINTER(Vector)),
]

coulomb_info = POINTER(struct_CoulombInfo)# install-dir/include/fock/fock.h: 36

# install-dir/include/fock/fock.h: 51
class struct_ExchangeInfo(Structure):
    pass

struct_ExchangeInfo.__slots__ = [
    'lr_frac',
    'sr_frac',
    'omega_',
    'type',
    'if_aux',
    'if_grids',
    'if_final',
    'batch_size',
    'K',
    'K_b',
    'mol',
    'aux',
    'aux_final',
    'aux2e_matrix_rev',
    'aux2e_matrix_rev_final',
    'o_matrix',
    'grids',
    'o_matrix_final',
    'grids_final',
    'grids_fname',
]
struct_ExchangeInfo._fields_ = [
    ('lr_frac', c_double),
    ('sr_frac', c_double),
    ('omega_', c_double),
    ('type', Kbuilder_type),
    ('if_aux', c_bool),
    ('if_grids', c_bool),
    ('if_final', c_bool),
    ('batch_size', c_int),
    ('K', POINTER(Matrix)),
    ('K_b', POINTER(Matrix)),
    ('mol', mol_info),
    ('aux', bas_info),
    ('aux_final', bas_info),
    ('aux2e_matrix_rev', Matrix),
    ('aux2e_matrix_rev_final', Matrix),
    ('o_matrix', Matrix),
    ('grids', grids_info),
    ('o_matrix_final', Matrix),
    ('grids_final', grids_info),
    ('grids_fname', String),
]

exchange_info = POINTER(struct_ExchangeInfo)# install-dir/include/fock/fock.h: 67

# install-dir/include/fock/dft.h: 42
class struct_DispInfo(Structure):
    pass

disp_info = POINTER(struct_DispInfo)# install-dir/include/fock/dft.h: 41

struct_DispInfo.__slots__ = [
    's6',
    's8',
    's10',
    'alp6',
    'alp8',
    'alp10',
    'rs6',
    'rs8',
    'rs10',
    'cab',
    'r0ab',
    'mxc',
    'atm_c',
    'mol',
    'func_type',
    'disp_type',
]
struct_DispInfo._fields_ = [
    ('s6', c_double),
    ('s8', c_double),
    ('s10', c_double),
    ('alp6', c_double),
    ('alp8', c_double),
    ('alp10', c_double),
    ('rs6', c_double),
    ('rs8', c_double),
    ('rs10', c_double),
    ('cab', Tensor4D * int(3)),
    ('r0ab', Matrix),
    ('mxc', List),
    ('atm_c', Vector),
    ('mol', mol_info),
    ('func_type', c_int),
    ('disp_type', c_int),
]

# install-dir/include/fock/dft.h: 60
class struct_DftInfo(Structure):
    pass

struct_DftInfo.__slots__ = [
    'xc_func',
    'xc_func_pol',
    'xc_frac',
    'func_num',
    'disp_type',
    'dft_type',
    'type',
    'dft_id',
    'max_type',
    'hf_frac',
    'grid_type',
    'grids',
    'mol',
    'disp',
    'dft_name',
]
struct_DftInfo._fields_ = [
    ('xc_func', POINTER(xc_func_type)),
    ('xc_func_pol', POINTER(xc_func_type)),
    ('xc_frac', POINTER(c_double)),
    ('func_num', c_int),
    ('disp_type', c_int),
    ('dft_type', c_int),
    ('type', POINTER(c_int)),
    ('dft_id', POINTER(c_int)),
    ('max_type', c_int),
    ('hf_frac', c_double),
    ('grid_type', GRIDS_TYPE),
    ('grids', grids_info),
    ('mol', mol_info),
    ('disp', disp_info),
    ('dft_name', c_int),
]

dft_info = POINTER(struct_DftInfo)# install-dir/include/fock/dft.h: 76

# install-dir/include/scf/hf.h: 38
class struct_anon_50(Structure):
    pass

struct_anon_50.__slots__ = [
    'max_iter',
    'max_delta_e',
    'max_delta_d',
    'max_fds_err',
]
struct_anon_50._fields_ = [
    ('max_iter', c_int),
    ('max_delta_e', c_double),
    ('max_delta_d', c_double),
    ('max_fds_err', c_double),
]

# install-dir/include/scf/hf.h: 84
class struct_anon_51(Structure):
    pass

struct_anon_51.__slots__ = [
    'charge_num',
    'charge_value',
    'ev_matrix',
    'e_ev',
    'e_nv',
    'e_tolv',
    'ev_grad',
]
struct_anon_51._fields_ = [
    ('charge_num', c_int),
    ('charge_value', Vector),
    ('ev_matrix', Matrix),
    ('e_ev', Vector),
    ('e_nv', Vector),
    ('e_tolv', Vector),
    ('ev_grad', Vector),
]

# install-dir/include/scf/hf.h: 10
class struct_HFInfo(Structure):
    pass

struct_HFInfo.__slots__ = [
    'open_type',
    'batch_size',
    'msize',
    'mol',
    'coulomb',
    'exchange',
    'dft',
    's_matrix',
    'x_matrix',
    's_half_matrix',
    't_matrix',
    'v_matrix',
    'h_matrix',
    'j_matrix',
    'k_matrix',
    'k_matrix_b',
    'f_matrix',
    'f_matrix_b',
    'c_matrix',
    'c_matrix_b',
    'd_matrix',
    'd_matrix_b',
    'xc_matrix',
    'xc_matrix_b',
    'ext_p',
    'alpha_egn',
    'beta_egn',
    'tol_energy',
    'alpha_seg',
    'beta_seg',
    'mult_seg',
    'charge_seg',
    'atm_seg',
    'e_t',
    'e_v',
    'e_j',
    'e_k',
    'e_xc',
    'nuc_rep',
    'S2',
    'ctrl',
    'diis_size',
    'diis_tr1',
    'diis_tr2',
    'if_sub_def',
    'if_sub_use',
    'sub_size',
    'subspace',
    's_matrix_sub',
    'x_matrix_sub',
    'f_matrix_sub',
    'f_matrix_b_sub',
    'c_matrix_sub',
    'c_matrix_b_sub',
    'd_matrix_sub',
    'd_matrix_b_sub',
    'if_aim_def',
    'bath_size',
    'aim_mu',
    'aim_enum',
    'bath_e',
    'bath_coup',
    's_matrix_aim',
    'x_matrix_aim',
    'f_matrix_aim',
    'f_matrix_b_aim',
    'c_matrix_aim',
    'c_matrix_b_aim',
    'd_matrix_aim',
    'd_matrix_b_aim',
    'background_charge',
    'jaux_fname',
    'kaux_fname',
    'jgrids_fname',
    'kgrids_fname',
    'jfgrids_fname',
    'kfgrids_fname',
]
struct_HFInfo._fields_ = [
    ('open_type', c_int),
    ('batch_size', c_int),
    ('msize', c_int),
    ('mol', mol_info),
    ('coulomb', coulomb_info),
    ('exchange', exchange_info),
    ('dft', dft_info),
    ('s_matrix', Matrix),
    ('x_matrix', Matrix),
    ('s_half_matrix', Matrix),
    ('t_matrix', Matrix),
    ('v_matrix', POINTER(Matrix)),
    ('h_matrix', POINTER(Matrix)),
    ('j_matrix', POINTER(Matrix)),
    ('k_matrix', POINTER(Matrix)),
    ('k_matrix_b', POINTER(Matrix)),
    ('f_matrix', POINTER(Matrix)),
    ('f_matrix_b', POINTER(Matrix)),
    ('c_matrix', POINTER(Matrix)),
    ('c_matrix_b', POINTER(Matrix)),
    ('d_matrix', POINTER(Matrix)),
    ('d_matrix_b', POINTER(Matrix)),
    ('xc_matrix', POINTER(Matrix)),
    ('xc_matrix_b', POINTER(Matrix)),
    ('ext_p', POINTER(Matrix)),
    ('alpha_egn', POINTER(Vector)),
    ('beta_egn', POINTER(Vector)),
    ('tol_energy', Vector),
    ('alpha_seg', List),
    ('beta_seg', List),
    ('mult_seg', List),
    ('charge_seg', List),
    ('atm_seg', List),
    ('e_t', Vector),
    ('e_v', Vector),
    ('e_j', Vector),
    ('e_k', Vector),
    ('e_xc', Vector),
    ('nuc_rep', Vector),
    ('S2', Vector),
    ('ctrl', struct_anon_50),
    ('diis_size', c_int),
    ('diis_tr1', c_double),
    ('diis_tr2', c_double),
    ('if_sub_def', c_bool),
    ('if_sub_use', c_bool),
    ('sub_size', c_int),
    ('subspace', Matrix),
    ('s_matrix_sub', Matrix),
    ('x_matrix_sub', Matrix),
    ('f_matrix_sub', POINTER(Matrix)),
    ('f_matrix_b_sub', POINTER(Matrix)),
    ('c_matrix_sub', POINTER(Matrix)),
    ('c_matrix_b_sub', POINTER(Matrix)),
    ('d_matrix_sub', POINTER(Matrix)),
    ('d_matrix_b_sub', POINTER(Matrix)),
    ('if_aim_def', c_bool),
    ('bath_size', c_int),
    ('aim_mu', c_double),
    ('aim_enum', c_double),
    ('bath_e', Vector),
    ('bath_coup', Matrix),
    ('s_matrix_aim', Matrix),
    ('x_matrix_aim', Matrix),
    ('f_matrix_aim', POINTER(Matrix)),
    ('f_matrix_b_aim', POINTER(Matrix)),
    ('c_matrix_aim', POINTER(Matrix)),
    ('c_matrix_b_aim', POINTER(Matrix)),
    ('d_matrix_aim', POINTER(Matrix)),
    ('d_matrix_b_aim', POINTER(Matrix)),
    ('background_charge', POINTER(struct_anon_51)),
    ('jaux_fname', String),
    ('kaux_fname', String),
    ('jgrids_fname', String),
    ('kgrids_fname', String),
    ('jfgrids_fname', String),
    ('kfgrids_fname', String),
]

hf_info = POINTER(struct_HFInfo)# install-dir/include/scf/hf.h: 99

# install-dir/include/input/input.h: 5
class struct_InpInfo(Structure):
    pass

struct_InpInfo.__slots__ = [
    'inpname',
    'mol_name',
    'dftfunc',
    'title',
    'basis_name',
    'aux_name',
    'j_grid_file',
    'k_grid_file',
    'k_grid_file_final',
    'hf_frac',
    'grid_type',
    'dft_frac',
    'dft_id',
    'ndft',
    'disp_type',
    'dft_name',
    'unit_bohr',
    'dovb',
    'dodft',
    'ihf_type',
    'iguess',
    'nstr',
    'nor',
    'nmul',
    'nao',
    'nae',
    'iroot',
    'nel',
    'itmax',
    'nb',
    'ncharge',
    'iscf',
    'boysloc',
    'dovbscf',
    'dobovb',
    'dovbcis',
    'dovbcisd',
    'dovbcids',
    'dovbpt2',
    'genstr',
    'dogopt',
    'dobfi',
    'fixc',
    'dopop',
    'dowfn',
    'orbtyp',
    'frgtyp',
    'wfntyp',
    'vbftyp',
    'dir2e',
    'inttyp',
    'ncor',
    'cicut',
    'inci',
    'ngroup',
    'nsav',
    'grpval',
    'strclass',
    'idxstate',
    'wstate',
    'wfn_name',
    'print_level',
    'DoGopt',
    'Gopt_Type',
    'Grad_Type',
    'Hess_Type',
    'Max_Gopt_Cycles',
    'doeda',
    'neda',
    'monomers',
    'doblw',
    'nblw',
]
struct_InpInfo._fields_ = [
    ('inpname', c_char * int(1024)),
    ('mol_name', c_char * int(1024)),
    ('dftfunc', c_char * int(40)),
    ('title', c_char * int(1024)),
    ('basis_name', c_char * int(4096)),
    ('aux_name', c_char * int(4096)),
    ('j_grid_file', String),
    ('k_grid_file', c_char * int(4096)),
    ('k_grid_file_final', c_char * int(4096)),
    ('hf_frac', c_double),
    ('grid_type', GRIDS_TYPE),
    ('dft_frac', POINTER(c_double)),
    ('dft_id', POINTER(c_int)),
    ('ndft', c_int),
    ('disp_type', c_int),
    ('dft_name', c_int),
    ('unit_bohr', c_int),
    ('dovb', c_int),
    ('dodft', c_int),
    ('ihf_type', c_int),
    ('iguess', c_int),
    ('nstr', c_int),
    ('nor', c_int),
    ('nmul', c_int),
    ('nao', c_int),
    ('nae', c_int),
    ('iroot', c_int),
    ('nel', c_int),
    ('itmax', c_int),
    ('nb', c_int),
    ('ncharge', c_int),
    ('iscf', c_int),
    ('boysloc', c_int),
    ('dovbscf', c_int),
    ('dobovb', c_int),
    ('dovbcis', c_int),
    ('dovbcisd', c_int),
    ('dovbcids', c_int),
    ('dovbpt2', c_int),
    ('genstr', c_int),
    ('dogopt', c_int),
    ('dobfi', c_int),
    ('fixc', c_int),
    ('dopop', c_int),
    ('dowfn', c_int),
    ('orbtyp', c_int),
    ('frgtyp', c_int),
    ('wfntyp', c_int),
    ('vbftyp', c_int),
    ('dir2e', c_int),
    ('inttyp', c_int),
    ('ncor', c_int),
    ('cicut', c_int),
    ('inci', c_int),
    ('ngroup', c_int),
    ('nsav', c_int),
    ('grpval', c_char * int(1024)),
    ('strclass', c_char * int(200)),
    ('idxstate', c_int * int(200)),
    ('wstate', c_double * int(200)),
    ('wfn_name', c_char * int(4096)),
    ('print_level', c_int),
    ('DoGopt', c_int),
    ('Gopt_Type', c_int),
    ('Grad_Type', c_int),
    ('Hess_Type', c_int),
    ('Max_Gopt_Cycles', c_int),
    ('doeda', c_int),
    ('neda', c_int),
    ('monomers', POINTER(c_int)),
    ('doblw', c_int),
    ('nblw', c_int),
]

inp_info = POINTER(struct_InpInfo)# install-dir/include/input/input.h: 53

# install-dir/include/parallel/para.h: 5
class struct_ParaInfo(Structure):
    pass

struct_ParaInfo.__slots__ = [
    'myproc',
    'nprocs',
    'ncores',
    'thread_num',
]
struct_ParaInfo._fields_ = [
    ('myproc', c_int),
    ('nprocs', c_int),
    ('ncores', c_int),
    ('thread_num', c_int),
]

para_info = POINTER(struct_ParaInfo)# install-dir/include/parallel/para.h: 9

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 41
class struct_VbInfo(Structure):
    pass

struct_VbInfo.__slots__ = [
    'ir',
    'iw',
    'isc',
    'ier',
    'ida',
    'ide',
    'ifo',
    'boysloc',
    'dovb',
    'dovbci',
    'dobfi',
    'dovbscf',
    'dobovb',
    'dovbcis',
    'dovbcisd',
    'dovbcids',
    'dovbpt2',
    'genstr',
    'dogopt',
    'orbtyp',
    'frgtyp',
    'wfntyp',
    'vbftyp',
    'orb_with_symm',
    'iguess',
    'iscf',
    'biovb',
    'itmax',
    'nel',
    'nmul',
    'nao',
    'nae',
    'nb',
    'natom',
    'npb',
    'vbsym_d',
    'moor',
    'mnor',
    'vbsym_igrd',
    'vbsym_ind',
    'nop',
    'nsymc',
    'nrep',
    'str_ori',
    'nvar',
    'indxcx',
    'indxxc',
    'ioor',
    'inor',
    'cvic',
    'nstr',
    'mxbond',
    'ndet',
    'nstr_scf',
    'ntstr',
    'strclass',
    'nor',
    'nor_scf',
    'nv',
    'ma0',
    'ma',
    'nvic',
    'dv',
    'nblock',
    'block_part_ov',
    'blocks',
    'noc_block',
    'mx_block',
    'nshell',
    'atm',
    'bas',
    'basidx',
    'env',
    'snorm',
    'n2e',
    'dir2e',
    'inttyp',
    'ssf',
    'hhf',
    'ggf',
    'xxf',
    'yyf',
    'zzf',
    'ekf',
    'g2eidx',
    'nb3num',
    'nb3idx',
    'hh',
    'ss',
    'gg',
    'xx',
    'yy',
    'zz',
    'ek',
    'ndetpair',
    'detpair',
    'idxpair',
    'iroot',
    'enuc',
    'eknuc',
    'xnuc',
    'ynuc',
    'znuc',
    'energy',
    'vbci_energy',
    'vbci_energy_davidson',
    'vbpt2_energy',
    'energyc',
    'epg',
    'gpg',
    'col',
    'hvb',
    'svb',
    'ncor',
    'cicut',
    'inci',
    'istr',
    'nbostr',
    'indxbovb',
    'inactbo',
    'nsav',
    'idxstate',
    'wstate',
    'state_energy',
    'ngroup',
    'readcoef',
    'grpidx',
    'grplist',
    'fixcol',
    'file_name',
    'aux_name',
    'k_grid_file',
    'k_grid_file_final',
    'dopop',
    'dowfn',
    'wfn_name',
    'gaaok',
    'ntgaa',
    'ngaa',
    'ntpar',
    'norg',
    'ntorg',
]
struct_VbInfo._fields_ = [
    ('ir', c_int),
    ('iw', c_int),
    ('isc', c_int),
    ('ier', c_int),
    ('ida', c_int),
    ('ide', c_int),
    ('ifo', c_int),
    ('boysloc', c_int),
    ('dovb', c_int),
    ('dovbci', c_int),
    ('dobfi', c_int),
    ('dovbscf', c_int),
    ('dobovb', c_int),
    ('dovbcis', c_int),
    ('dovbcisd', c_int),
    ('dovbcids', c_int),
    ('dovbpt2', c_int),
    ('genstr', c_int),
    ('dogopt', c_int),
    ('orbtyp', c_int),
    ('frgtyp', c_int),
    ('wfntyp', c_int),
    ('vbftyp', c_int),
    ('orb_with_symm', c_int),
    ('iguess', c_int),
    ('iscf', c_int),
    ('biovb', c_int),
    ('itmax', c_int),
    ('nel', c_int),
    ('nmul', c_int),
    ('nao', c_int),
    ('nae', c_int),
    ('nb', c_int),
    ('natom', c_int),
    ('npb', c_int),
    ('vbsym_d', c_int),
    ('moor', c_int),
    ('mnor', c_int),
    ('vbsym_igrd', c_int),
    ('vbsym_ind', c_int),
    ('nop', c_int),
    ('nsymc', POINTER(c_int)),
    ('nrep', POINTER(c_int)),
    ('str_ori', POINTER(c_int)),
    ('nvar', c_int),
    ('indxcx', POINTER(c_int)),
    ('indxxc', POINTER(c_int)),
    ('ioor', POINTER(c_int)),
    ('inor', POINTER(c_int)),
    ('cvic', POINTER(c_double)),
    ('nstr', c_int),
    ('mxbond', c_int),
    ('ndet', c_int),
    ('nstr_scf', c_int),
    ('ntstr', POINTER(c_int)),
    ('strclass', c_char * int(200)),
    ('nor', c_int),
    ('nor_scf', c_int),
    ('nv', POINTER(c_int)),
    ('ma0', POINTER(c_int)),
    ('ma', POINTER(c_int)),
    ('nvic', POINTER(c_int)),
    ('dv', POINTER(c_double)),
    ('nblock', c_int),
    ('block_part_ov', c_int),
    ('blocks', POINTER(c_int)),
    ('noc_block', POINTER(c_int)),
    ('mx_block', POINTER(c_int)),
    ('nshell', c_int),
    ('atm', POINTER(c_int)),
    ('bas', POINTER(c_int)),
    ('basidx', POINTER(c_int)),
    ('env', POINTER(c_double)),
    ('snorm', POINTER(c_double)),
    ('n2e', c_int),
    ('dir2e', c_int),
    ('inttyp', c_int),
    ('ssf', POINTER(c_double)),
    ('hhf', POINTER(c_double)),
    ('ggf', POINTER(c_double)),
    ('xxf', POINTER(c_double)),
    ('yyf', POINTER(c_double)),
    ('zzf', POINTER(c_double)),
    ('ekf', POINTER(c_double)),
    ('g2eidx', POINTER(c_int)),
    ('nb3num', POINTER(c_int)),
    ('nb3idx', POINTER(c_int)),
    ('hh', POINTER(c_double)),
    ('ss', POINTER(c_double)),
    ('gg', POINTER(c_double)),
    ('xx', POINTER(c_double)),
    ('yy', POINTER(c_double)),
    ('zz', POINTER(c_double)),
    ('ek', POINTER(c_double)),
    ('ndetpair', c_int),
    ('detpair', POINTER(c_int)),
    ('idxpair', POINTER(c_int)),
    ('iroot', c_int),
    ('enuc', c_double),
    ('eknuc', c_double),
    ('xnuc', c_double),
    ('ynuc', c_double),
    ('znuc', c_double),
    ('energy', c_double),
    ('vbci_energy', c_double),
    ('vbci_energy_davidson', c_double),
    ('vbpt2_energy', c_double),
    ('energyc', c_double),
    ('epg', c_double),
    ('gpg', c_double),
    ('col', POINTER(c_double)),
    ('hvb', POINTER(c_double)),
    ('svb', POINTER(c_double)),
    ('ncor', c_int),
    ('cicut', c_int),
    ('inci', c_int),
    ('istr', POINTER(c_int)),
    ('nbostr', POINTER(c_int)),
    ('indxbovb', POINTER(c_int)),
    ('inactbo', POINTER(c_int)),
    ('nsav', c_int),
    ('idxstate', c_int * int(200)),
    ('wstate', c_double * int(200)),
    ('state_energy', c_double * int(200)),
    ('ngroup', c_int),
    ('readcoef', c_int),
    ('grpidx', POINTER(c_int)),
    ('grplist', POINTER(c_int)),
    ('fixcol', POINTER(c_double)),
    ('file_name', c_char * int(4096)),
    ('aux_name', c_char * int(4096)),
    ('k_grid_file', c_char * int(4096)),
    ('k_grid_file_final', c_char * int(4096)),
    ('dopop', c_int),
    ('dowfn', c_int),
    ('wfn_name', c_char * int(4096)),
    ('gaaok', c_int),
    ('ntgaa', (c_int * int(10100)) * int(10)),
    ('ngaa', c_int * int(10)),
    ('ntpar', POINTER(c_int)),
    ('norg', c_int),
    ('ntorg', c_int * int(21)),
]

vb_info = POINTER(struct_VbInfo)# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 155

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 157
for _lib in _libs.values():
    if not _lib.has("init_vb_param", "cdecl"):
        continue
    init_vb_param = _lib.get("init_vb_param", "cdecl")
    init_vb_param.argtypes = [mol_info, inp_info, vb_info]
    init_vb_param.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 158
for _lib in _libs.values():
    if not _lib.has("del_vb_str", "cdecl"):
        continue
    del_vb_str = _lib.get("del_vb_str", "cdecl")
    del_vb_str.argtypes = [vb_info]
    del_vb_str.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 160
for _lib in _libs.values():
    if not _lib.has("expand_str_orb", "cdecl"):
        continue
    expand_str_orb = _lib.get("expand_str_orb", "cdecl")
    expand_str_orb.argtypes = [POINTER(c_char * int(1024)), POINTER(c_int), c_int, POINTER(c_int)]
    expand_str_orb.restype = None
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 162
for _lib in _libs.values():
    if not _lib.has("readinp_vb", "cdecl"):
        continue
    readinp_vb = _lib.get("readinp_vb", "cdecl")
    readinp_vb.argtypes = [mol_info, inp_info, vb_info, para_info, String]
    readinp_vb.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 163
for _lib in _libs.values():
    if not _lib.has("vbprep", "cdecl"):
        continue
    vbprep = _lib.get("vbprep", "cdecl")
    vbprep.argtypes = [hf_info, inp_info, vb_info, para_info]
    vbprep.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 165
for _lib in _libs.values():
    if not _lib.has("readstr", "cdecl"):
        continue
    readstr = _lib.get("readstr", "cdecl")
    readstr.argtypes = [String, POINTER(c_int), POINTER(c_double), c_int, c_int, c_int]
    readstr.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 166
for _lib in _libs.values():
    if not _lib.has("getstr", "cdecl"):
        continue
    getstr = _lib.get("getstr", "cdecl")
    getstr.argtypes = [String, vb_info]
    getstr.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 167
for _lib in _libs.values():
    if not _lib.has("genstr", "cdecl"):
        continue
    genstr = _lib.get("genstr", "cdecl")
    genstr.argtypes = [vb_info]
    genstr.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 168
for _lib in _libs.values():
    if not _lib.has("getfrg", "cdecl"):
        continue
    getfrg = _lib.get("getfrg", "cdecl")
    getfrg.argtypes = [String, mol_info, vb_info]
    getfrg.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 169
for _lib in _libs.values():
    if not _lib.has("getorb", "cdecl"):
        continue
    getorb = _lib.get("getorb", "cdecl")
    getorb.argtypes = [String, vb_info]
    getorb.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 170
for _lib in _libs.values():
    if not _lib.has("readorb", "cdecl"):
        continue
    readorb = _lib.get("readorb", "cdecl")
    readorb.argtypes = [String, vb_info]
    readorb.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 171
for _lib in _libs.values():
    if not _lib.has("detect_blocks", "cdecl"):
        continue
    detect_blocks = _lib.get("detect_blocks", "cdecl")
    detect_blocks.argtypes = [vb_info]
    detect_blocks.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 172
for _lib in _libs.values():
    if not _lib.has("getvars", "cdecl"):
        continue
    getvars = _lib.get("getvars", "cdecl")
    getvars.argtypes = [vb_info]
    getvars.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 174
for _lib in _libs.values():
    if not _lib.has("vbguess", "cdecl"):
        continue
    vbguess = _lib.get("vbguess", "cdecl")
    vbguess.argtypes = [hf_info, inp_info, vb_info]
    vbguess.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 175
for _lib in _libs.values():
    if not _lib.has("vb_autoguess", "cdecl"):
        continue
    vb_autoguess = _lib.get("vb_autoguess", "cdecl")
    vb_autoguess.argtypes = [hf_info, vb_info]
    vb_autoguess.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 176
for _lib in _libs.values():
    if not _lib.has("vb_unitguess", "cdecl"):
        continue
    vb_unitguess = _lib.get("vb_unitguess", "cdecl")
    vb_unitguess.argtypes = [vb_info]
    vb_unitguess.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 177
for _lib in _libs.values():
    if not _lib.has("vb_readguess", "cdecl"):
        continue
    vb_readguess = _lib.get("vb_readguess", "cdecl")
    vb_readguess.argtypes = [inp_info, vb_info]
    vb_readguess.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 178
for _lib in _libs.values():
    if not _lib.has("vb_monboguess", "cdecl"):
        continue
    vb_monboguess = _lib.get("vb_monboguess", "cdecl")
    vb_monboguess.argtypes = [inp_info, vb_info]
    vb_monboguess.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 179
for _lib in _libs.values():
    if not _lib.has("cvitra", "cdecl"):
        continue
    cvitra = _lib.get("cvitra", "cdecl")
    cvitra.argtypes = [POINTER(c_double), POINTER(c_int), POINTER(c_int), POINTER(c_double), c_int, c_int]
    cvitra.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 181
for _lib in _libs.values():
    if not _lib.has("vbscf", "cdecl"):
        continue
    vbscf = _lib.get("vbscf", "cdecl")
    vbscf.argtypes = [hf_info, inp_info, vb_info, c_int, c_int]
    vbscf.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 182
for _lib in _libs.values():
    if not _lib.has("normalize", "cdecl"):
        continue
    normalize = _lib.get("normalize", "cdecl")
    normalize.argtypes = [c_int, c_int, POINTER(c_double), POINTER(c_int), POINTER(c_int), POINTER(c_double)]
    normalize.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 183
for _lib in _libs.values():
    if not _lib.has("gendetpair", "cdecl"):
        continue
    gendetpair = _lib.get("gendetpair", "cdecl")
    gendetpair.argtypes = [vb_info, c_int, c_int, c_int]
    gendetpair.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 184
for _lib in _libs.values():
    if not _lib.has("str2det", "cdecl"):
        continue
    str2det = _lib.get("str2det", "cdecl")
    str2det.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_int), c_int, c_int, c_int, c_int]
    str2det.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 185
for _lib in _libs.values():
    if not _lib.has("parput", "cdecl"):
        continue
    parput = _lib.get("parput", "cdecl")
    parput.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int), c_int, c_int, c_int, c_int]
    parput.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 186
for _lib in _libs.values():
    if not _lib.has("parget", "cdecl"):
        continue
    parget = _lib.get("parget", "cdecl")
    parget.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int), c_int, c_int, c_int, c_int]
    parget.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 187
for _lib in _libs.values():
    if not _lib.has("invmat", "cdecl"):
        continue
    invmat = _lib.get("invmat", "cdecl")
    invmat.argtypes = [POINTER(c_double), POINTER(c_double), c_int, c_int, POINTER(c_double)]
    invmat.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 188
for _lib in _libs.values():
    if not _lib.has("mppinvmat", "cdecl"):
        continue
    mppinvmat = _lib.get("mppinvmat", "cdecl")
    mppinvmat.argtypes = [POINTER(c_double), c_int, c_int]
    mppinvmat.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 189
for _lib in _libs.values():
    if not _lib.has("orbout", "cdecl"):
        continue
    orbout = _lib.get("orbout", "cdecl")
    orbout.argtypes = [POINTER(c_double), POINTER(c_int), POINTER(c_int), POINTER(c_double), c_double, c_double, POINTER(c_double), POINTER(c_double), c_int, c_int, c_int, c_int]
    orbout.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 190
for _lib in _libs.values():
    if not _lib.has("get_ngto", "cdecl"):
        continue
    get_ngto = _lib.get("get_ngto", "cdecl")
    get_ngto.argtypes = [mol_info]
    get_ngto.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 192
for _lib in _libs.values():
    if not _lib.has("int_data_trans", "cdecl"):
        continue
    int_data_trans = _lib.get("int_data_trans", "cdecl")
    int_data_trans.argtypes = [mol_info, POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_double), c_int]
    int_data_trans.restype = None
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 193
for _lib in _libs.values():
    if not _lib.has("cal_1eint", "cdecl"):
        continue
    cal_1eint = _lib.get("cal_1eint", "cdecl")
    cal_1eint.argtypes = [vb_info]
    cal_1eint.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 194
for _lib in _libs.values():
    if not _lib.has("cal_2eint", "cdecl"):
        continue
    cal_2eint = _lib.get("cal_2eint", "cdecl")
    cal_2eint.argtypes = [vb_info]
    cal_2eint.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 195
for _lib in _libs.values():
    if not _lib.has("build_shidx", "cdecl"):
        continue
    build_shidx = _lib.get("build_shidx", "cdecl")
    build_shidx.argtypes = [vb_info, POINTER(c_double), POINTER(c_int), POINTER(c_int)]
    build_shidx.restype = None
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 196
for _lib in _libs.values():
    if not _lib.has("lab", "cdecl"):
        continue
    lab = _lib.get("lab", "cdecl")
    lab.argtypes = [c_int, c_int]
    lab.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 197
for _lib in _libs.values():
    if not _lib.has("diag", "cdecl"):
        continue
    diag = _lib.get("diag", "cdecl")
    diag.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int, c_int]
    diag.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 198
for _lib in _libs.values():
    if not _lib.has("detect_ndb", "cdecl"):
        continue
    detect_ndb = _lib.get("detect_ndb", "cdecl")
    detect_ndb.argtypes = [POINTER(c_int), c_int, c_int, c_int, c_int]
    detect_ndb.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 199
for _lib in _libs.values():
    if not _lib.has("init_bovb", "cdecl"):
        continue
    init_bovb = _lib.get("init_bovb", "cdecl")
    init_bovb.argtypes = [vb_info]
    init_bovb.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 201
for _lib in _libs.values():
    if not _lib.has("tdm_vbscf", "cdecl"):
        continue
    tdm_vbscf = _lib.get("tdm_vbscf", "cdecl")
    tdm_vbscf.argtypes = [hf_info, vb_info, c_int]
    tdm_vbscf.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 202
for _lib in _libs.values():
    if not _lib.has("vblbfgs", "cdecl"):
        continue
    vblbfgs = _lib.get("vblbfgs", "cdecl")
    vblbfgs.argtypes = [hf_info, vb_info, c_int]
    vblbfgs.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 203
for _lib in _libs.values():
    if not _lib.has("derdifb", "cdecl"):
        continue
    derdifb = _lib.get("derdifb", "cdecl")
    derdifb.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_int), POINTER(c_int), c_int, c_int, c_int, POINTER(c_double), hf_info, vb_info]
    derdifb.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 205
for _lib in _libs.values():
    if not _lib.has("rdm_vbscf", "cdecl"):
        continue
    rdm_vbscf = _lib.get("rdm_vbscf", "cdecl")
    rdm_vbscf.argtypes = [vb_info, c_int]
    rdm_vbscf.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 206
for _lib in _libs.values():
    if not _lib.has("gradient_rdm", "cdecl"):
        continue
    gradient_rdm = _lib.get("gradient_rdm", "cdecl")
    gradient_rdm.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), vb_info]
    gradient_rdm.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 208
for _lib in _libs.values():
    if not _lib.has("compute_vb", "cdecl"):
        continue
    compute_vb = _lib.get("compute_vb", "cdecl")
    compute_vb.argtypes = [hf_info, inp_info, vb_info, c_int, c_int]
    compute_vb.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 209
for _lib in _libs.values():
    if not _lib.has("boysloc", "cdecl"):
        continue
    boysloc = _lib.get("boysloc", "cdecl")
    boysloc.argtypes = [vb_info, c_int]
    boysloc.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 210
for _lib in _libs.values():
    if not _lib.has("gen_virtual_orb", "cdecl"):
        continue
    gen_virtual_orb = _lib.get("gen_virtual_orb", "cdecl")
    gen_virtual_orb.argtypes = [vb_info, c_int]
    gen_virtual_orb.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 211
for _lib in _libs.values():
    if not _lib.has("gen_vbci_str", "cdecl"):
        continue
    gen_vbci_str = _lib.get("gen_vbci_str", "cdecl")
    gen_vbci_str.argtypes = [vb_info, c_int]
    gen_vbci_str.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 212
for _lib in _libs.values():
    if not _lib.has("vbci", "cdecl"):
        continue
    vbci = _lib.get("vbci", "cdecl")
    vbci.argtypes = [vb_info, c_int]
    vbci.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 213
for _lib in _libs.values():
    if not _lib.has("vbpt2", "cdecl"):
        continue
    vbpt2 = _lib.get("vbpt2", "cdecl")
    vbpt2.argtypes = [vb_info, c_int]
    vbpt2.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 214
for _lib in _libs.values():
    if not _lib.has("clean_vb_modules", "cdecl"):
        continue
    clean_vb_modules = _lib.get("clean_vb_modules", "cdecl")
    clean_vb_modules.argtypes = [vb_info]
    clean_vb_modules.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 215
for _lib in _libs.values():
    if not _lib.has("eigencalc", "cdecl"):
        continue
    eigencalc = _lib.get("eigencalc", "cdecl")
    eigencalc.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_double, c_int, vb_info]
    eigencalc.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 216
for _lib in _libs.values():
    if not _lib.has("calc_den", "cdecl"):
        continue
    calc_den = _lib.get("calc_den", "cdecl")
    calc_den.argtypes = [POINTER(c_double), POINTER(c_double), vb_info]
    calc_den.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 217
for _lib in _libs.values():
    if not _lib.has("calc_nos", "cdecl"):
        continue
    calc_nos = _lib.get("calc_nos", "cdecl")
    calc_nos.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_int), POINTER(c_double), c_int]
    calc_nos.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 218
for _lib in _libs.values():
    if not _lib.has("schmidt0", "cdecl"):
        continue
    schmidt0 = _lib.get("schmidt0", "cdecl")
    schmidt0.argtypes = [c_int, c_int, c_int, c_int, POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int]
    schmidt0.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 219
for _lib in _libs.values():
    if not _lib.has("schmidt1", "cdecl"):
        continue
    schmidt1 = _lib.get("schmidt1", "cdecl")
    schmidt1.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int, c_int, c_int]
    schmidt1.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 220
for _lib in _libs.values():
    if not _lib.has("xtra1", "cdecl"):
        continue
    xtra1 = _lib.get("xtra1", "cdecl")
    xtra1.argtypes = [vb_info, c_int]
    xtra1.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 221
for _lib in _libs.values():
    if not _lib.has("vbdet", "cdecl"):
        continue
    vbdet = _lib.get("vbdet", "cdecl")
    vbdet.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_int), POINTER(c_int), c_int, c_int, vb_info]
    vbdet.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 222
for _lib in _libs.values():
    if not _lib.has("build_nb3idx", "cdecl"):
        continue
    build_nb3idx = _lib.get("build_nb3idx", "cdecl")
    build_nb3idx.argtypes = [vb_info]
    build_nb3idx.restype = c_int
    break

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 8
try:
    GEN_TYP = 0
except:
    pass

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 9
try:
    HAO_TYP = 1
except:
    pass

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 10
try:
    BDO_TYP = 2
except:
    pass

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 11
try:
    OEO_TYP = 3
except:
    pass

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 13
try:
    FRG_ATM = 0
except:
    pass

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 14
try:
    FRG_SAO = 1
except:
    pass

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 16
try:
    WFN_STR = 0
except:
    pass

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 17
try:
    WFN_DET = 1
except:
    pass

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 19
try:
    VBF_DET = 0
except:
    pass

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 20
try:
    VBF_PPD = 1
except:
    pass

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 22
try:
    GUS_AUTO = 0
except:
    pass

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 23
try:
    GUS_UNIT = 1
except:
    pass

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 24
try:
    GUS_READ = 2
except:
    pass

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 25
try:
    GUS_RDCI = 3
except:
    pass

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 26
try:
    GUS_MO = 4
except:
    pass

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 27
try:
    GUS_NBO = 5
except:
    pass

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 29
try:
    TDM_SCF = 2
except:
    pass

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 30
try:
    RDM_SCF = 5
except:
    pass

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 32
try:
    INT_CINT = 0
except:
    pass

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 33
try:
    INT_XINT = 1
except:
    pass

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 35
try:
    INT_TOL = 1e-10
except:
    pass

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 36
try:
    SVD_TOL = 1e-5
except:
    pass

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 38
try:
    MAX_STATE = 200
except:
    pass

# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 39
try:
    MAX_PATH = 4096
except:
    pass

VbInfo = struct_VbInfo# /home/xiatao/xmvb/install-dir/include/vb/vb.h: 41

# No inserted files

# No prefix-stripping

