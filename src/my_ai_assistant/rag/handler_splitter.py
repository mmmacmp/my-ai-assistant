from typing import Any, Callable, Literal

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# class HandlerRecursiveCharacterTextSplitter(RecursiveCharacterTextSplitter):
#     def __init__(
#         self,
#         handler: Callable[
#             [str, list[str]],
#             list[str],
#         ] | None = None,
#         *args,
#         **kwargs,
#     ) -> None:
#         super().__init__(*args, **kwargs)
#         self.handler = (
#             handler
#             if handler is not None
#             else lambda _, chunks: chunks
#         )

#     def split_text(self, text: str) -> list[str]:
#         chunks = super().split_text(text)
#         return self.handler(text, chunks)

class HandlerRecursiveCharacterTextSplitter(RecursiveCharacterTextSplitter):

    def __init__(self, handler: Callable[[str, list[str]], list[str]] | None = None, separators: list[str] | None = None, keep_separator: bool | Literal['start'] | Literal['end'] = True, is_separator_regex: bool = False, **kwargs: Any) -> None:
        super().__init__(separators, keep_separator, is_separator_regex, **kwargs)
        self.handler = handler

    def split_text(self, text: str) -> list[str]:
        chunks = super().split_text(text)
        if self.handler != None:
            chunks_after_handler = self.handler(text, chunks)
            return chunks_after_handler
        return chunks