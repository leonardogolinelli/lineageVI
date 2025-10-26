# LineageVI Documentation

This directory contains the Sphinx documentation for LineageVI.

## Building the Documentation

### Prerequisites

Install the required packages:

```bash
pip install -r requirements.txt
```

### Build Commands

```bash
# Build HTML documentation
make html

# Build and serve with auto-reload (for development)
make livehtml

# Build PDF documentation
make pdf

# Build EPUB documentation
make epub

# Check for broken links
make linkcheck

# Run doctests
make doctest
```

### Development

For development, use the live HTML build:

```bash
make livehtml
```

This will start a local server at `http://localhost:8000` with auto-reload enabled.

## Documentation Structure

- `index.rst` - Main documentation page
- `installation.rst` - Installation instructions
- `quickstart.rst` - Quick start guide
- `api/` - API reference documentation
- `tutorials/` - Detailed tutorials
- `examples/` - Example analyses
- `_static/` - Static files (CSS, images, etc.)
- `_templates/` - Custom templates

## Adding New Documentation

1. Create new `.rst` files in the appropriate directory
2. Add them to the relevant `index.rst` file
3. Rebuild the documentation with `make html`

## Configuration

The main configuration is in `conf.py`. Key settings:

- `html_theme = 'sphinx_rtd_theme'` - Uses Read the Docs theme
- `extensions` - List of Sphinx extensions
- `autodoc_default_options` - Settings for automatic documentation generation

## Deployment

The documentation can be deployed to:

- **GitHub Pages**: Use the `sphinx.ext.githubpages` extension
- **Read the Docs**: Connect your repository to Read the Docs
- **Local hosting**: Build and serve the HTML files

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure the LineageVI package is installed
2. **Missing modules**: Check that all dependencies are installed
3. **Build errors**: Check the Sphinx output for specific error messages

### Getting Help

- Check the [Sphinx documentation](https://www.sphinx-doc.org/)
- Look at the build output for specific error messages
- Ensure all required packages are installed
