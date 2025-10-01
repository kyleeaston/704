# import pefile

# def extract_features(filepath):
#     try:
#         pe = pefile.PE(filepath)
#         features = [
#             pe.OPTIONAL_HEADER.SizeOfCode,
#             len(pe.sections),
#             pe.OPTIONAL_HEADER.AddressOfEntryPoint,
#             pe.OPTIONAL_HEADER.SizeOfImage
#         ]
#         return features
#     except Exception as e:
#         print(f"Error parsing {filepath}: {e}")
#         return None

# pe_features_lief.py
import math
import re
from typing import Tuple, List

import numpy as np

# prefer lief if available
try:
    import lief
    LIEF_AVAILABLE = True
except Exception:
    LIEF_AVAILABLE = False

import pefile  # fallback

# ---------------------------
# Utilities
# ---------------------------
def shannon_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    freq = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probs = freq / freq.sum()
    return float(-np.sum([p * math.log2(p) for p in probs if p > 0]))


_ASCII_PRINTABLE_RE = re.compile(rb'[\x20-\x7e]{4,}')
_UNICODE_PRINTABLE_RE = re.compile((rb'(?:[\x20-\x7e]\x00){4,}'))


def extract_strings(data: bytes) -> Tuple[list, list]:
    ascii_matches = [m.decode('latin1', errors='ignore') for m in _ASCII_PRINTABLE_RE.findall(data)]
    unicode_matches = []
    for m in _UNICODE_PRINTABLE_RE.findall(data):
        try:
            unicode_matches.append(m.decode('utf-16le', errors='ignore'))
        except Exception:
            continue
    return ascii_matches, unicode_matches


def byte_histogram(data: bytes) -> np.ndarray:
    if not data:
        return np.zeros(256, dtype=np.float32)
    hist = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256).astype(np.float32)
    return hist / hist.sum()


TOP_APIS = [
    "CreateRemoteThread", "VirtualAlloc", "VirtualProtect", "VirtualAllocEx", "WriteProcessMemory",
    "ReadProcessMemory", "GetProcAddress", "LoadLibrary", "GetModuleHandle", "CreateProcess",
    "OpenProcess", "SetThreadContext", "ResumeThread", "SuspendThread", "NtCreateFile",
    "RegOpenKeyEx", "InternetOpenUrl", "URLDownloadToFile", "WinExec", "ShellExecute",
    "WSAStartup", "socket", "connect", "send", "recv", "CreateFile", "ReadFile", "WriteFile",
    "DeviceIoControl", "AdjustTokenPrivileges", "OpenSCManager", "StartService", "SetWindowsHookEx"
]


class PEFeatureExtractorLief:
    def __init__(self, section_entropy_bins: int = 10):
        self.section_entropy_bins = section_entropy_bins

        # base names
        self._base_feature_names = [
            "file_size",
            "virtual_size",
            "number_of_sections",
            "size_of_code",
            "size_of_initialized_data",
            "size_of_uninitialized_data",
            "entry_point",
            "subsystem",
            "dll_characteristics",
            "number_of_rva_and_sizes",
        ]

        # extra LIEF features
        self._lief_feature_names = [
            "has_signature",        # boolean
            "overlay_size",         # bytes of overlay (data after last section)
            "machine_type",         # numeric
            "compile_timestamp",    # timestamp from header
        ]

        self._section_summary_names = [
            "mean_section_entropy",
            "max_section_entropy",
            "min_section_entropy",
            "mean_section_raw_size",
            "max_section_raw_size",
            "min_section_raw_size",
            "executable_section_count",
            "readable_section_count",
            "writable_section_count",
            "avg_section_alignment",
        ]

        self._imp_exp_res_names = [
            "num_imported_dlls",
            "num_imported_functions",
            "num_exported_functions",
            "num_resources",
            "total_resource_size",
        ]

        self._string_feature_names = [
            "num_ascii_strings",
            "num_unicode_strings",
            "avg_ascii_string_len",
            "avg_unicode_string_len",
            "count_urls_in_strings",
            "count_cmd_like_strings",
        ]

        self._section_entropy_bin_names = [f"sec_entropy_bin_{i}" for i in range(self.section_entropy_bins)]
        self._api_presence_names = [f"api_present_{api}" for api in TOP_APIS]
        self._byte_hist_names = [f"byte_hist_{i}" for i in range(256)]

        self.feature_names = (
            self._base_feature_names
            + self._lief_feature_names
            + self._section_summary_names
            + self._imp_exp_res_names
            + self._string_feature_names
            + self._section_entropy_bin_names
            + self._api_presence_names
            + self._byte_hist_names
        )

    def extract(self, data: bytes) -> Tuple[np.ndarray, List[str]]:
        file_size = len(data)
        # defaults
        virtual_size = 0
        number_of_sections = 0
        size_of_code = 0
        size_of_initialized_data = 0
        size_of_uninitialized_data = 0
        entry_point = 0
        subsystem = 0
        dll_characteristics = 0
        number_of_rva_and_sizes = 0

        has_signature = 0.0
        overlay_size = 0.0
        machine_type = 0.0
        compile_timestamp = 0.0

        section_entropies = []
        section_sizes = []
        exec_count = read_count = write_count = 0
        avg_section_alignment = 0.0

        num_imported_dlls = num_imported_functions = num_exported_functions = 0
        num_resources = total_resource_size = 0

        ascii_strings = []
        unicode_strings = []
        imported_api_names = []

        # try LIEF first
        lief_ok = False
        if LIEF_AVAILABLE:
            try:
                # lief expects a bytes-like; parse returns a Binary
                binary = lief.parse(list(data))
                if binary and binary.format == lief.EXE_FORMATS.PE:
                    lief_ok = True

                    # header & optional header
                    machine_type = float(getattr(binary.header, "machine", 0) or getattr(binary.header, "machine_type", 0) or 0)
                    compile_timestamp = float(getattr(binary.header, "time_date_stamp", 0) or 0)

                    # optional header-ish fields (LIEF names vary)
                    oh = getattr(binary, "optional_header", None)
                    if oh:
                        size_of_code = float(getattr(oh, "size_of_code", 0) or 0)
                        size_of_initialized_data = float(getattr(oh, "size_of_initialized_data", 0) or 0)
                        size_of_uninitialized_data = float(getattr(oh, "size_of_uninitialized_data", 0) or 0)
                        entry_point = float(getattr(oh, "address_of_entrypoint", 0) or getattr(oh, "addressof_entrypoint", 0) or 0)
                        subsystem = float(getattr(oh, "subsystem", 0) or 0)
                        dll_characteristics = float(getattr(oh, "dll_characteristics", 0) or 0)
                        number_of_rva_and_sizes = float(getattr(oh, "numberof_rva_and_sizes", 0) or 0)

                    # sections via LIEF
                    sections = getattr(binary, "sections", []) or []
                    number_of_sections = len(sections)
                    for s in sections:
                        try:
                            raw = bytes(s.content) if getattr(s, "content", None) is not None else b""
                        except Exception:
                            raw = b""
                        ent = shannon_entropy(raw)
                        section_entropies.append(ent)
                        section_sizes.append(len(raw))

                        flags = getattr(s, "characteristics", 0) or 0
                        # IMAGE_SCN_MEM_EXECUTE = 0x20000000, READ=0x40000000, WRITE=0x80000000
                        if flags & 0x20000000:
                            exec_count += 1
                        if flags & 0x40000000:
                            read_count += 1
                        if flags & 0x80000000:
                            write_count += 1

                        try:
                            align = (getattr(s, "virtual_size", 0) or 0) / (getattr(s, "size", 0) or 1)
                        except Exception:
                            align = 0
                        avg_section_alignment += align

                    if number_of_sections:
                        avg_section_alignment /= number_of_sections

                    # imports via LIEF
                    try:
                        imports = getattr(binary, "imports", []) or []
                        num_imported_dlls = float(len(imports))
                        imp_count = 0
                        for entry in imports:
                            # entry.name and entry.entries (list of imported symbols)
                            for e in getattr(entry, "entries", []) or []:
                                name = getattr(e, "name", None)
                                if name:
                                    imported_api_names.append(str(name))
                                    imp_count += 1
                        num_imported_functions = float(imp_count)
                    except Exception:
                        num_imported_dlls = num_imported_functions = 0.0

                    # exports
                    try:
                        exports = getattr(binary, "export", None)
                        if exports:
                            num_exported_functions = float(len(getattr(exports, "entries", []) or []))
                        else:
                            num_exported_functions = 0.0
                    except Exception:
                        num_exported_functions = 0.0

                    # resources via LIEF (counts/sizes)
                    try:
                        resources = getattr(binary, "resources", None)
                        if resources:
                            # LIEF resources object structure can vary; try to approximate
                            # Use binary.get_overlay or binary.has_overlay
                            # We'll approximate resource counting via raw directories if possible
                            # Fallback to 0 if not accessible.
                            # LIEF doesn't offer a simple total size in all versions
                            num_resources = float(len(getattr(resources, "entries", []) or []))
                            total_resource_size = 0.0  # leave 0 for now to be safe
                        else:
                            num_resources = 0.0
                            total_resource_size = 0.0
                    except Exception:
                        num_resources = total_resource_size = 0.0

                    # signature and overlay
                    try:
                        has_signature = 1.0 if getattr(binary, "has_signature", False) else 0.0
                    except Exception:
                        has_signature = 0.0

                    try:
                        overlay_size = float(getattr(binary, "overlay", None) and len(getattr(binary, "overlay", b"")) or 0.0)
                    except Exception:
                        # LIEF has binary.has_overlay() / binary.overlay
                        try:
                            if hasattr(binary, "has_overlay") and binary.has_overlay:
                                overlay_size = float(len(binary.overlay or b""))
                            else:
                                overlay_size = 0.0
                        except Exception:
                            overlay_size = 0.0

            except Exception:
                lief_ok = False

        # If LIEF parsing failed or not available, fallback to pefile for the same values
        if not lief_ok:
            try:
                pe = pefile.PE(data=data)
            except Exception:
                # Cannot parse -> return mostly zeros but include file_size and byte_hist
                byte_hist = byte_histogram(data)
                vec = np.zeros(len(self.feature_names), dtype=np.float32)
                vec[0] = float(file_size)
                vec[-256:] = byte_hist
                return vec, self.feature_names

            # header / optional header via pefile
            try:
                virtual_size = float(sum(s.Misc_VirtualSize for s in pe.sections) if pe.sections else 0)
            except Exception:
                virtual_size = 0.0

            number_of_sections = float(len(pe.sections) if pe.sections else 0)
            size_of_code = float(getattr(pe.OPTIONAL_HEADER, "SizeOfCode", 0) or 0)
            size_of_initialized_data = float(getattr(pe.OPTIONAL_HEADER, "SizeOfInitializedData", 0) or 0)
            size_of_uninitialized_data = float(getattr(pe.OPTIONAL_HEADER, "SizeOfUninitializedData", 0) or 0)
            entry_point = float(getattr(pe.OPTIONAL_HEADER, "AddressOfEntryPoint", 0) or 0)
            subsystem = float(getattr(pe.OPTIONAL_HEADER, "Subsystem", 0) or 0)
            dll_characteristics = float(getattr(pe.OPTIONAL_HEADER, "DllCharacteristics", 0) or 0)
            number_of_rva_and_sizes = float(getattr(pe.OPTIONAL_HEADER, "NumberOfRvaAndSizes", 0) or 0)

            # pefile sections
            exec_count = read_count = write_count = 0
            avg_section_alignment = 0.0
            section_entropies = []
            section_sizes = []
            for s in getattr(pe, "sections", []):
                try:
                    raw = s.get_data()
                except Exception:
                    raw = b""
                ent = shannon_entropy(raw)
                section_entropies.append(ent)
                section_sizes.append(len(raw))
                characteristics = getattr(s, "Characteristics", 0) or 0
                if characteristics & 0x20000000:
                    exec_count += 1
                if characteristics & 0x40000000:
                    read_count += 1
                if characteristics & 0x80000000:
                    write_count += 1
                try:
                    align = (s.Misc_VirtualSize / (s.SizeOfRawData or 1))
                except Exception:
                    align = 0
                avg_section_alignment += align

            if number_of_sections:
                avg_section_alignment = avg_section_alignment / number_of_sections
            else:
                avg_section_alignment = 0.0

            # imports via pefile
            try:
                if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
                    num_imported_dlls = float(len(pe.DIRECTORY_ENTRY_IMPORT))
                    imp_count = 0
                    for entry in pe.DIRECTORY_ENTRY_IMPORT:
                        for imp in getattr(entry, "imports", []) or []:
                            if getattr(imp, "name", None):
                                imported_api_names.append(imp.name.decode("latin1", errors="ignore"))
                                imp_count += 1
                    num_imported_functions = float(imp_count)
            except Exception:
                num_imported_dlls = num_imported_functions = 0.0

            # exports
            try:
                if hasattr(pe, "DIRECTORY_ENTRY_EXPORT") and pe.DIRECTORY_ENTRY_EXPORT:
                    num_exported_functions = float(len(pe.DIRECTORY_ENTRY_EXPORT.symbols))
                else:
                    num_exported_functions = 0.0
            except Exception:
                num_exported_functions = 0.0

            # resources
            try:
                if hasattr(pe, "DIRECTORY_ENTRY_RESOURCE") and pe.DIRECTORY_ENTRY_RESOURCE:
                    # naive counting
                    num_resources = float(len(pe.DIRECTORY_ENTRY_RESOURCE.entries))
                    total_resource_size = 0.0
                else:
                    num_resources = 0.0
                    total_resource_size = 0.0
            except Exception:
                num_resources = total_resource_size = 0.0

            # signature detection via pefile is harder; leave as 0
            has_signature = 0.0

            # overlay: last section offset + raw size -> overlay bytes
            try:
                last = pe.sections[-1]
                overlay_start = last.PointerToRawData + last.SizeOfRawData
                if overlay_start < len(data):
                    overlay_size = float(len(data) - overlay_start)
                else:
                    overlay_size = 0.0
            except Exception:
                overlay_size = 0.0

            try:
                machine_type = float(getattr(pe.FILE_HEADER, "Machine", 0) or 0)
            except Exception:
                machine_type = 0.0

            try:
                compile_timestamp = float(getattr(pe.FILE_HEADER, "TimeDateStamp", 0) or 0)
            except Exception:
                compile_timestamp = 0.0

            # strings from whole file
            ascii_strings, unicode_strings = extract_strings(data)

        # If lief succeeded, we may not have filled ascii_strings etc. — ensure strings exist
        if not ascii_strings or not unicode_strings:
            try:
                ascii_strings, unicode_strings = extract_strings(data)
            except Exception:
                ascii_strings, unicode_strings = [], []

        # string heuristics
        num_ascii = float(len(ascii_strings))
        num_unicode = float(len(unicode_strings))
        avg_ascii_len = float(np.mean([len(s) for s in ascii_strings]) if ascii_strings else 0.0)
        avg_unicode_len = float(np.mean([len(s) for s in unicode_strings]) if unicode_strings else 0.0)
        urls_count = float(sum(1 for s in ascii_strings if ("http://" in s or "https://" in s or "://" in s)))
        cmd_like_count = float(sum(1 for s in ascii_strings if any(k in s.lower() for k in ("cmd.exe", "powershell", "schtasks", "at.exe"))))

        # section entropy summary
        if section_entropies:
            mean_section_entropy = float(np.mean(section_entropies))
            max_section_entropy = float(np.max(section_entropies))
            min_section_entropy = float(np.min(section_entropies))
            mean_section_size = float(np.mean(section_sizes))
            max_section_size = float(np.max(section_sizes))
            min_section_size = float(np.min(section_sizes))
        else:
            mean_section_entropy = max_section_entropy = min_section_entropy = 0.0
            mean_section_size = max_section_size = min_section_size = 0.0

        # entropy bins
        sec_entropy_hist = np.histogram(section_entropies, bins=self.section_entropy_bins, range=(0.0, 8.0))[0].astype(np.float32)
        if sec_entropy_hist.sum() > 0:
            sec_entropy_hist = sec_entropy_hist / sec_entropy_hist.sum()
        else:
            sec_entropy_hist = sec_entropy_hist.astype(np.float32)

        # api presence vector
        imported_lower = [n.lower() for n in imported_api_names]
        api_presence = [1.0 if any(api.lower() in name for name in imported_lower) else 0.0 for api in TOP_APIS]

        # byte hist
        byte_hist = byte_histogram(data).astype(np.float32)

        # assemble
        values = []
        values += [
            float(file_size),
            float(virtual_size),
            float(number_of_sections),
            float(size_of_code),
            float(size_of_initialized_data),
            float(size_of_uninitialized_data),
            float(entry_point),
            float(subsystem),
            float(dll_characteristics),
            float(number_of_rva_and_sizes),
        ]

        values += [
            float(has_signature),
            float(overlay_size),
            float(machine_type),
            float(compile_timestamp),
        ]

        values += [
            mean_section_entropy,
            max_section_entropy,
            min_section_entropy,
            mean_section_size,
            max_section_size,
            min_section_size,
            float(exec_count),
            float(read_count),
            float(write_count),
            float(avg_section_alignment),
        ]

        values += [
            float(num_imported_dlls),
            float(num_imported_functions),
            float(num_exported_functions),
            float(num_resources),
            float(total_resource_size),
        ]

        values += [
            float(num_ascii),
            float(num_unicode),
            float(avg_ascii_len),
            float(avg_unicode_len),
            float(urls_count),
            float(cmd_like_count),
        ]

        values += sec_entropy_hist.tolist()
        values += api_presence
        values += byte_hist.tolist()

        vec = np.array(values, dtype=np.float32)

        # ensure proper length
        if vec.shape[0] != len(self.feature_names):
            desired = len(self.feature_names)
            cur = vec.shape[0]
            if cur < desired:
                pad = np.zeros(desired - cur, dtype=np.float32)
                vec = np.concatenate([vec, pad])
            else:
                vec = vec[:desired]

        return vec, self.feature_names
