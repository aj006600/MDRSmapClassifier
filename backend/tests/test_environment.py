"""驗證容器裡裝出來的環境符合版本鎖定。

這個測試要在 backend 的 dev stage 容器裡跑,在別的環境過了不代表容器裡裝得起來。
"""

import platform
import warnings


def test_pyedm_imports_at_pinned_version_on_linux():
    # pyEDM 1.x 在 import 時用到 pkg_resources,會發出 DeprecationWarning;
    # 能 import 成功本身就證明 setuptools 的版本上限有效
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import pyEDM

    assert platform.system() == "Linux"
    # 上游另有 1.15.4.1,所以完全比對,不用前綴
    assert pyEDM.__version__ == "1.15.4.0"
