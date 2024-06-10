import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
   
#with open("requirements.txt") as f:
#    requirements = f.read().splitlines()

setuptools.setup(name='guap',
      version='1.0.0',
      author='Jordan Patracone',
      author_email='jordan.frecon.deloire@univ-st-etienne.fr',
      description='Generalized Universal Adversarial Perturbations',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/JordanFrecon/guap',
      license='MIT',
      package_dir={"": "src"},
      packages=setuptools.find_packages(where="src"),
      python_requires=">=3.6",
      #install_requires=requirements,
      )

