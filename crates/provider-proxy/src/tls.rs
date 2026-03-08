// Code is from https://github.com/hatoo/http-mitm-proxy/blob/7c8c3bde77173af6385d5d0ffaea6105498df1ff/src/tls.rs (MIT-licensed)

#[derive(Debug, Clone)]
pub struct CertifiedKeyDer {
    pub cert_der: Vec<u8>,
    /// Pkcs8
    pub key_der: Vec<u8>,
}

pub fn generate_cert(
    host: String,
    root_cert: &rcgen::Issuer<'_, rcgen::KeyPair>,
) -> CertifiedKeyDer {
    let mut cert_params = rcgen::CertificateParams::new(vec![host.clone()]).unwrap();
    cert_params
        .key_usages
        .push(rcgen::KeyUsagePurpose::DigitalSignature);
    cert_params
        .extended_key_usages
        .push(rcgen::ExtendedKeyUsagePurpose::ServerAuth);
    cert_params
        .extended_key_usages
        .push(rcgen::ExtendedKeyUsagePurpose::ClientAuth);
    cert_params.distinguished_name = {
        let mut dn = rcgen::DistinguishedName::new();
        dn.push(rcgen::DnType::CommonName, host);
        dn
    };

    let key_pair = rcgen::KeyPair::generate().unwrap();

    let cert = cert_params.signed_by(&key_pair, root_cert).unwrap();

    CertifiedKeyDer {
        cert_der: cert.der().to_vec(),
        key_der: key_pair.serialize_der(),
    }
}
