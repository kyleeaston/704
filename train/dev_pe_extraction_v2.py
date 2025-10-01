from typing import Tuple, List
import numpy as np
import lief
import math

def compute_entropy(data: bytes) -> float:
    """Compute Shannon entropy of a byte array."""
    if not data:
        return 0.0
    # Count occurrences of each byte (0-255)
    freq = [0] * 256
    for b in data:
        freq[b] += 1
    entropy = 0.0
    length = len(data)
    for f in freq:
        if f > 0:
            p = f / length
            entropy -= p * math.log2(p)
    return entropy


SUSPICIOUS_APIS = {
    "VirtualAlloc",
    "VirtualProtect",
    "LoadLibraryA",
    "LoadLibraryW",
    "GetProcAddress",
    "CreateRemoteThread",
    "WinExec",
    "WriteProcessMemory",
    "InternetOpenA",
    "InternetOpenUrlA",
}


class PEFeatureExtractorLief:
    def extract(self, data: bytes) -> Tuple[np.ndarray, List[str]]:
        # Try to parse the PE; if it fails, we'll still return zeros later
        try:
            pe = lief.parse(list(data))
        except Exception:
            pe = None

        features = []
        feature_names = []

        # 1. File size
        try:
            file_size = len(data)
        except Exception:
            file_size = 0
        features.append(file_size)
        feature_names.append("file_size")

        # 1a. File entropy
        try:
            entropy = compute_entropy(data)
        except Exception:
            entropy = 0.0
        features.append(entropy)
        feature_names.append("file_entropy")

        # 2. Number of sections
        try:
            n_sections = len(pe.sections) if pe and pe.sections else 0
        except Exception:
            n_sections = 0
        features.append(n_sections)
        feature_names.append("num_sections")

        # 3. Entry point RVA
        try:
            entry_point = pe.entrypoint if pe and pe.entrypoint else 0
        except Exception:
            entry_point = 0
        features.append(entry_point)
        feature_names.append("entry_point_rva")

        # 4. Size of .text section
        try:
            text_size = 0
            if pe and pe.sections:
                for sec in pe.sections:
                    sec_name = (
                        sec.name.decode(errors="ignore")
                        if isinstance(sec.name, bytes)
                        else sec.name
                    )
                    if ".text" in sec_name.lower():
                        text_size = sec.size
                        break
        except Exception:
            text_size = 0
        features.append(text_size)
        feature_names.append("text_section_size")

        # 5. Number of imported functions
        try:
            n_imports = 0
            if pe and hasattr(pe, "imports"):
                for imp in pe.imports:
                    n_imports += len(imp.entries)
        except Exception:
            n_imports = 0
        features.append(n_imports)
        feature_names.append("num_imported_functions")\
        
        # 6. Multi-hot vector for suspicious APIs
        api_vector = np.zeros(len(SUSPICIOUS_APIS), dtype=np.float32)
        for idx, api_name in enumerate(SUSPICIOUS_APIS):
            try:
                if pe and hasattr(pe, "imports"):
                    for imp in pe.imports:
                        for entry in imp.entries:
                            name = entry.name or ""
                            if name == api_name:
                                api_vector[idx] = 1.0
                                raise StopIteration  # Stop searching once found
            except StopIteration:
                continue
            except Exception:
                # If any error occurs for this API, leave 0
                continue

        # Append multi-hot API features
        features.extend(api_vector)
        feature_names.extend([f"import_{name}" for name in SUSPICIOUS_APIS])

        return np.array(features, dtype=np.float32), feature_names
