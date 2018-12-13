from setuptools import setup

def main():
  setup(
      name='best',
      version='0.2',
      packages=['best', 'best.models', 'best.solvers', 'best.abstraction', 'best.logic'],
      license='BSD-3',
      author='Petter Nilsson',
      author_email='pettni@caltech.edu',
      description='Routines for formal abstraction and controller synthesis',
      package_data={'best.logic': ['logic/binaries/mac/scheck2','logic/binaries/mac/ltl2ba']}
  )

if __name__ == '__main__':
  main()