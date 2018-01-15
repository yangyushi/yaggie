from distutils.core import setup

setup(name='yaggie',
      version='0.1',
      packages=['yaggie',
                'yaggie.engine',
                'yaggie.engine.colloids',
                'yaggie.engine.linking',
                'yaggie.engine.mytrack'],
      py_modules=['yaggie.analysis',
                  'yaggie.render',
                  'yaggie.utility']
      )
