#!/bin/bash

# Create directory and download certificates
mkdir -p /usr/local/share/ca-certificates/Yandex
wget "https://storage.yandexcloud.net/cloud-certs/RootCA.pem" \
    -O /usr/local/share/ca-certificates/Yandex/RootCA.crt
wget "https://storage.yandexcloud.net/cloud-certs/IntermediateCA.pem" \
    -O /usr/local/share/ca-certificates/Yandex/IntermediateCA.crt

# Set permissions
chmod 655 /usr/local/share/ca-certificates/Yandex/RootCA.crt \
    /usr/local/share/ca-certificates/Yandex/IntermediateCA.crt

# Update CA certificates
update-ca-certificates

# Execute the CMD from Dockerfile
exec "$@"
