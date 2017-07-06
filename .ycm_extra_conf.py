def FlagsForFile( filename, **kwargs ):
      return {'flags': [ '-x', '-Wall', '-Wextra', '-Werror', '-I/opt/rocm/include','-I/opt/rocm/hip/include']}
