import numpy as np
import pytest

from turban.process.generic.api import Processing


@pytest.fixture
def check_vocabulary_in_xarray():
    """Function ``(processing, vocabulary)`` that checks the variable attributes
    of ``processing.to_xarray()`` against ``vocabulary``, see
    ``_check_vocabulary_in_xarray``."""

    def _check_vocabulary_in_xarray(p: Processing, vocabulary: list[dict]):
        """Check that the attributes from ``vocabulary`` end up in ``p.to_xarray()``.

        For every variable (data variable or coordinate) of every level node that is
        in any of the vocabularies, all attributes of the merged vocabulary entry
        (later vocabularies take precedence) must be set. Fails if no exported variable
        is covered by the vocabulary at all.

        Parameters
        ----------
        p : Processing
            Processing object to export with its default vocabulary.
        vocabulary : list of dict
            Expected vocabularies, ``{varname: {attribute: value}}``.
        """
        tree = p.to_xarray()
        n_checked = 0
        for level_name, node in tree.children.items():
            ds = node.to_dataset()
            for varname in ds.variables:
                expected = {}
                for v in vocabulary:
                    expected.update(v.get(varname, {}))
                for attr, value in expected.items():
                    assert (
                        attr in ds[varname].attrs
                    ), f"{level_name}/{varname}: attribute {attr!r} missing"
                    np.testing.assert_equal(
                        ds[varname].attrs[attr],
                        value,
                        err_msg=f"{level_name}/{varname}: attribute {attr!r}",
                    )
                n_checked += bool(expected)
        assert n_checked > 0, "no exported variable is covered by the vocabulary"

    return _check_vocabulary_in_xarray
