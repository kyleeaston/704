import pefile

def extract_features(filepath):
    try:
        pe = pefile.PE(filepath)
        features = [
            pe.OPTIONAL_HEADER.SizeOfCode,
            len(pe.sections),
            pe.OPTIONAL_HEADER.AddressOfEntryPoint,
            pe.OPTIONAL_HEADER.SizeOfImage
        ]
        return features
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None