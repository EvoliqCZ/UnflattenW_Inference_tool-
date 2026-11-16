from __future__ import annotations

import math
from collections.abc import Iterator

from scanner_state import Operand

_WHITESPACE = b"\x00\t\n\x0c\r "
_DELIMITERS = b"()<>[]{}/%"
_SIMPLE_ESCAPES = {
    ord("n"): 0x0A, ord("r"): 0x0D, ord("t"): 0x09,
    ord("b"): 0x08, ord("f"): 0x0C, ord("("): 0x28,
    ord(")"): 0x29, ord("\\"): 0x5C,
}


class ContentLexer:
    """Tokenize PDF content. String operands keep only their decoded byte
    codes (never the human-readable text) so a caller can look up glyph
    widths by code; no other PDF content ever surfaces."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.index = 0

    def __iter__(self) -> Iterator[str | Operand]:
        while True:
            item = self._next()
            if item is None:
                return
            yield item

    def skip_inline_image(self) -> None:
        while True:
            item = self._next()
            if item is None:
                raise ValueError("inline image dictionary is unterminated")
            if item == "ID":
                break
        if self.index >= len(self.data) or self.data[self.index] not in _WHITESPACE:
            raise ValueError("inline image data separator is absent")
        self.index += 1
        end = self._inline_image_end()
        if end is None:
            raise ValueError("inline image data is unterminated")
        self.index = end

    def _inline_image_end(self) -> int | None:
        index = self.index
        while index + 2 <= len(self.data):
            found = self.data.find(b"EI", index)
            if found < 0:
                return None
            before = self.data[found - 1] if found else ord(" ")
            after = self.data[found + 2] if found + 2 < len(self.data) else ord(" ")
            if before in _WHITESPACE and after in _WHITESPACE + _DELIMITERS:
                return found + 2
            index = found + 2
        return None

    def _next(self) -> str | Operand | None:
        self._skip_space_and_comments()
        if self.index >= len(self.data):
            return None
        byte = self.data[self.index]
        if byte == ord("("):
            return self._literal_string()
        if byte == ord("["):
            return self._array()
        if byte == ord("<"):
            if self.data[self.index : self.index + 2] == b"<<":
                return self._dictionary()
            return self._hex_string()
        if byte == ord("/"):
            return self._name()
        if byte in b"]>)":
            raise ValueError("unexpected PDF delimiter")
        return self._bare()

    def _skip_space_and_comments(self) -> None:
        while self.index < len(self.data):
            if self.data[self.index] in _WHITESPACE:
                self.index += 1
            elif self.data[self.index] == ord("%"):
                newline = self.data.find(b"\n", self.index)
                self.index = len(self.data) if newline < 0 else newline + 1
            else:
                return

    def _literal_string(self) -> Operand:
        """Decode escapes per ISO 32000-1 7.3.4.2 into byte codes (for glyph
        width lookup only -- never assembled into text or exposed)."""
        self.index += 1
        depth = 1
        codes = bytearray()
        while self.index < len(self.data):
            byte = self.data[self.index]
            if byte == ord("\\"):
                self.index += 1
                if self.index >= len(self.data):
                    raise ValueError("unterminated PDF literal string")
                decoded, consumed = self._literal_escape()
                self.index += consumed
                if decoded is not None:
                    codes.append(decoded)
                continue
            if byte == ord("("):
                depth += 1
            elif byte == ord(")"):
                depth -= 1
                if not depth:
                    self.index += 1
                    return Operand("string", len(codes), bytes(codes))
            if byte in (0x0D, 0x0A):
                codes.append(0x0A)
                if byte == 0x0D and self.index + 1 < len(self.data) \
                        and self.data[self.index + 1] == 0x0A:
                    self.index += 1
            else:
                codes.append(byte)
            self.index += 1
        raise ValueError("unterminated PDF literal string")

    def _literal_escape(self) -> tuple[int | None, int]:
        """Byte after a backslash: (decoded value or None, bytes consumed)."""
        byte = self.data[self.index]
        if byte in _SIMPLE_ESCAPES:
            return _SIMPLE_ESCAPES[byte], 1
        if byte in (0x0D, 0x0A):  # line continuation: no byte emitted
            crlf = byte == 0x0D and self.index + 1 < len(self.data) \
                and self.data[self.index + 1] == 0x0A
            return None, 2 if crlf else 1
        if ord("0") <= byte <= ord("7"):
            digits = bytes((byte,))
            end = self.index + 1
            while len(digits) < 3 and end < len(self.data) \
                    and ord("0") <= self.data[end] <= ord("7"):
                digits += bytes((self.data[end],))
                end += 1
            return int(digits, 8) & 0xFF, end - self.index
        return byte, 1  # backslash before any other byte: that byte literally

    def _hex_string(self) -> Operand:
        end = self.data.find(b">", self.index + 1)
        if end < 0:
            raise ValueError("unterminated PDF hex string")
        payload = self.data[self.index + 1 : end]
        if any(byte not in b"0123456789abcdefABCDEF" + _WHITESPACE for byte in payload):
            raise ValueError("invalid PDF hex string")
        self.index = end + 1
        digits = bytes(byte for byte in payload if byte not in _WHITESPACE)
        if len(digits) % 2:
            digits += b"0"
        codes = bytes.fromhex(digits.decode("ascii"))
        return Operand("string", len(codes), codes)

    def _array(self) -> Operand:
        self.index += 1
        values: list[Operand] = []
        while True:
            self._skip_space_and_comments()
            if self.index >= len(self.data):
                raise ValueError("unterminated PDF array")
            if self.data[self.index] == ord("]"):
                self.index += 1
                return Operand("array", tuple(values))
            item = self._next()
            if not isinstance(item, Operand):
                raise ValueError("operator inside PDF array")
            values.append(item)

    def _dictionary(self) -> Operand:
        self.index += 2
        depth = 1
        while self.index < len(self.data) and depth:
            if self.data[self.index : self.index + 2] == b"<<":
                depth += 1
                self.index += 2
            elif self.data[self.index : self.index + 2] == b">>":
                depth -= 1
                self.index += 2
            elif self.data[self.index] == ord("("):
                self._literal_string()
            else:
                self.index += 1
        if depth:
            raise ValueError("unterminated PDF dictionary")
        return Operand("dictionary")

    def _name(self) -> Operand:
        end = self.index + 1
        while end < len(self.data) and self.data[end] not in _WHITESPACE + _DELIMITERS:
            end += 1
        value = self.data[self.index : end].decode("latin-1")
        self.index = end
        return Operand("name", value)

    def _bare(self) -> str | Operand:
        end = self.index + 1
        while end < len(self.data) and self.data[end] not in _WHITESPACE + _DELIMITERS:
            end += 1
        value = self.data[self.index : end].decode("latin-1")
        self.index = end
        if value in {"true", "false"}:
            return Operand("boolean", value == "true")
        if value == "null":
            return Operand("null")
        try:
            number = float(value)
        except ValueError:
            return value
        return Operand("number", number if math.isfinite(number) else None)
