/** A file the user picked, tagged with the kind it will submit as (`api/upload.ts`). */
export interface SelectedFile {
  id: string
  file: File
  kind: 'pdf' | 'image'
}

export function selectedFileKind(file: File): SelectedFile['kind'] {
  return file.type === 'application/pdf' ? 'pdf' : 'image'
}
