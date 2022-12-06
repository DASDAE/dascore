"""
Create the html tables for parameters and such from dataframes.
"""
import inspect
from functools import cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


@cache
def get_env():
    """Get the template environment."""
    template_path = Path(__file__).absolute().parent / '_templates'
    env = Environment(loader=FileSystemLoader(template_path))
    return env


@cache
def get_template(name):
    """Get the template for rendering tables."""
    env = get_env()
    template = env.get_template(name)
    return template


def build_table(df, caption=None):
    """
    An opinionated function to make a dataframe into html table.
    """
    template = get_template('table.html')
    columns = [x.capitalize() for x in df.columns]
    rows = df.to_records(index=False).tolist()
    out = template.render(columns=columns, rows=rows, caption=caption)
    return out


def build_signature(data):
    """Return html of signature block."""

    def get_params(sig):
        params = [str(x).replace("'", "") for x in sig.parameters.values()]
        return params

    def get_return_line(sig):
        return_an = sig.return_annotation
        if return_an is inspect._empty:
            return_str = ')'
        else:
            return_str = f')-> {return_an}'
        return return_str


    def get_sig_dict(sig, data):
        """Create a dict of render-able signature stuff."""
        # TODO: add links
        out = dict(
            params=get_params(sig),
            return_line=get_return_line(sig),
            data_type=data['data_type'],
            name=data['name'],
        )
        return out

    # no need to do anything if entity is not callable.
    sig = data['signature']
    if not sig:
        return ''

    template = get_template('signature.html')
    sig_dict = get_sig_dict(sig, data)
    out = template.render(**sig_dict)
    return out
