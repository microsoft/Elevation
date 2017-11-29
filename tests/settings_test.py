import unittest
import sys

from mock import patch, Mock, PropertyMock, MagicMock


class SettingsTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(SettingsTest, self).__init__(*args, **kwargs)

    # @unittest.skip("ignore")
    def test_settings_mock(self):
        sys.stdout = sys.__stdout__

        from elevation import settings, prediction_pipeline, load_data
        from elevation.cmds import fit, predict
        import elevation

        mock_cachedir = "MOCK_CACHE"
        cachedir_patch = patch('elevation.settings.cachedir',  mock_cachedir)
        cachedir_patch.start()
        assert mock_cachedir == settings.cachedir
        assert mock_cachedir == prediction_pipeline.settings.cachedir
        assert mock_cachedir == load_data.settings.cachedir
        assert mock_cachedir == fit.settings.cachedir
        assert mock_cachedir == predict.settings.cachedir
        assert mock_cachedir == elevation.settings.cachedir
        assert mock_cachedir == elevation.prediction_pipeline.settings.cachedir
        assert mock_cachedir == elevation.load_data.settings.cachedir
        assert mock_cachedir == elevation.cmds.fit.settings.cachedir
        assert mock_cachedir == elevation.cmds.predict.settings.cachedir
        cachedir_patch.stop()

        mock_tmpdir = "MOCK_TMP"
        tmpdir_patch = patch('elevation.settings.tmpdir', mock_tmpdir)
        tmpdir_patch.start()
        assert mock_tmpdir == settings.tmpdir
        assert mock_tmpdir == prediction_pipeline.settings.tmpdir
        assert mock_tmpdir == load_data.settings.tmpdir
        assert mock_tmpdir == fit.settings.tmpdir
        assert mock_tmpdir == predict.settings.tmpdir
        assert mock_tmpdir == elevation.settings.tmpdir
        assert mock_tmpdir == elevation.prediction_pipeline.settings.tmpdir
        assert mock_tmpdir == elevation.load_data.settings.tmpdir
        assert mock_tmpdir == elevation.cmds.fit.settings.tmpdir
        assert mock_tmpdir == elevation.cmds.predict.settings.tmpdir
        tmpdir_patch.stop()
