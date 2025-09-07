def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    for b in bytestring:
        print(b)
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])


print(decode_utf8_bytes_to_str_wrong("hello".encode("utf-8")))
