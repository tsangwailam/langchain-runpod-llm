'''
langchain-runpod-llm | setup.py
Called to setup the runpod-python package.
'''

from setuptools import setup, find_packages

# README.md > long_description
with open('README.md', encoding='utf-8') as long_description_file:
    long_description = long_description_file.read()

# requirements.txt > requirements
with open('requirements.txt', encoding="UTF-8") as requirements_file:
    install_requires = requirements_file.read().splitlines()


if __name__ == "__main__":
    setup(
        # the name must match the folder name 'verysimplemodule'
        name="langchain-runpod-llm",
        use_scm_version=True,
        author="William Tsang",
        author_email="<contact@williamtsang.me>",
        description="Package to use Runpod LLM API endpoint",
        long_description=long_description,
        long_description_content_type='text/markdown',
        # install_requires=install_requires,
        packages=find_packages(),
        url='https://github.com/tsangwailam/langchain-runpod-llm',
        project_urls={
            'Documentation': 'https://github.com/tsangwailam/langchain-runpod-llm/blob/main/README.md',
            'Source': 'https://github.com/tsangwailam/langchain-runpod-llm',
            'Bug Tracker': 'https://github.com/tsangwailam/langchain-runpod-llm/issues',
            'Changelog': 'https://github.com/tsangwailam/langchain-runpod-llm/blob/main/CHANGELOG.md'
        },
        include_package_data=True,
        license='MIT',

        keywords=['python', 'langchain', 'runpod', 'llm', 'llama2'],
        classifiers=[
            'Environment :: Web Environment',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Topic :: Internet :: WWW/HTTP',
            'Topic :: Internet :: WWW/HTTP :: Dynamic Content'
        ]
    )
