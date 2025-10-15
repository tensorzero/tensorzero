#!/bin/bash

# Generate an extremely long-lived self-signed certificate for OTLP Collector
# Valid for 100 years

set -e

# Configuration
CERT_DIR="${1:-.}"
CERT_NAME="${2:-otlp-collector}"
DAYS=36500  # 100 years
KEY_SIZE=2048

# Create directory if it doesn't exist
mkdir -p "$CERT_DIR"

# Certificate file paths
CERT_FILE="$CERT_DIR/$CERT_NAME.crt"
KEY_FILE="$CERT_DIR/$CERT_NAME.key"
CSR_FILE="$CERT_DIR/$CERT_NAME.csr"

echo "Generating self-signed certificate valid for $DAYS days (~100 years)..."

# Generate private key
openssl genrsa -out "$KEY_FILE" $KEY_SIZE

# Generate certificate signing request
openssl req -new \
  -key "$KEY_FILE" \
  -out "$CSR_FILE" \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

# Generate self-signed certificate
openssl x509 -req \
  -days $DAYS \
  -in "$CSR_FILE" \
  -signkey "$KEY_FILE" \
  -out "$CERT_FILE" \
  -extfile <(printf "subjectAltName=DNS:localhost,DNS:*.localhost,IP:127.0.0.1")

# Clean up CSR file
rm "$CSR_FILE"

echo "Certificate generated successfully!"
echo "Certificate: $CERT_FILE"
echo "Private Key: $KEY_FILE"
echo ""
echo "Certificate details:"
openssl x509 -in "$CERT_FILE" -text -noout | grep -E "(Subject|Issuer|Not Before|Not After|Public-Key)"