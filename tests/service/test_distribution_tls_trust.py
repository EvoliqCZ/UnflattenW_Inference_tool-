import httpx


def test_model_download_tls_does_not_trust_compromised_emsign_chain():
    context = httpx.create_ssl_context()
    names = {
        name
        for cert in context.get_ca_certs()
        for group in cert["subject"]
        for kind, name in group
        if kind == "commonName"
    }
    assert "emSign ECC Root CA - G3" not in names
