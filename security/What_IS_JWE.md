---
parent: Security
layout: home
nav_order: 8
name: JWE
---

# What is JWE?
JSON Web Encryption (JWE) is a standard for securely transmitting data as a JSON object. It provides confidentiality by encrypting the payload, ensuring that only authorized parties can access the information.

## HTTPs vs JWE
While HTTPS secures data in transit using TLS, JWE adds an additional layer of security by encrypting the data itself. This is particularly useful for scenarios where data needs to be stored securely or transmitted through untrusted channels.

Also, HTTPS may not protect data in an end-to-end manner, especially in micro service architecture where data may pass through multiple intermediaries. JWE ensures that the data remains encrypted and secure throughout its lifecycle.

Key point to note here is that JWE can be used in conjunction with HTTPS to provide a more robust security solution that address complex integration scenarios.

