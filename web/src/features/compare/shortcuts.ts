type ShortcutLabelKey =
  | 'compare.shortcuts.page'
  | 'compare.shortcuts.firstLast'
  | 'compare.shortcuts.zoomStep'
  | 'compare.shortcuts.fit'
  | 'compare.shortcuts.actualSize'
  | 'compare.shortcuts.mode'
  | 'compare.shortcuts.diffToggle'
  | 'compare.shortcuts.opacityStep'
  | 'compare.shortcuts.lock'
  | 'compare.shortcuts.maximizeToggle'
  | 'compare.shortcuts.help'

/** UX §5.9: the comparison viewer's single-key shortcut table, keyed by the exact `KeyboardEvent.key`. */
export interface ShortcutRow {
  /** Human-readable key combo shown in `ShortcutHelpDialog`, not matched against events. */
  keys: string
  /** A literal union, not `string` -- `t()` is typed strictly against `sk.json`'s key set (design §7.5). */
  labelKey: ShortcutLabelKey
}

export const SHORTCUT_ROWS: ShortcutRow[] = [
  { keys: '← / →', labelKey: 'compare.shortcuts.page' },
  { keys: 'Home / End', labelKey: 'compare.shortcuts.firstLast' },
  { keys: '+ / −', labelKey: 'compare.shortcuts.zoomStep' },
  { keys: '0', labelKey: 'compare.shortcuts.fit' },
  { keys: '1', labelKey: 'compare.shortcuts.actualSize' },
  { keys: 'S / O / W', labelKey: 'compare.shortcuts.mode' },
  { keys: 'D', labelKey: 'compare.shortcuts.diffToggle' },
  { keys: '[ / ]', labelKey: 'compare.shortcuts.opacityStep' },
  { keys: 'L', labelKey: 'compare.shortcuts.lock' },
  { keys: 'F', labelKey: 'compare.shortcuts.maximizeToggle' },
  { keys: '?', labelKey: 'compare.shortcuts.help' },
]
