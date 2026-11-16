// design §6.2 / UX §8.1 / H8: served unhashed from the `dist/` root with
// `Cache-Control: no-cache`, loaded as a blocking, non-module <script src>
// from <head> before the app bundle. Sets the `dark` class on <html>
// synchronously so there is no theme flash. Deliberately plain, dependency-
// free JS: `public/` files are copied verbatim by Vite, not bundled, so this
// cannot import `src/lib/theme.ts` -- the decision logic (preference ->
// effective theme) is intentionally kept identical and covered by
// `theme-init.test.ts`, which loads and evaluates this exact file.
;(function () {
  var STORAGE_KEY = 'unflatten-theme'
  var stored = null
  try {
    stored = window.localStorage.getItem(STORAGE_KEY)
  } catch (e) {
    stored = null
  }
  var preference = stored === 'light' || stored === 'dark' || stored === 'system' ? stored : 'system'
  var systemPrefersDark =
    typeof window.matchMedia === 'function' && window.matchMedia('(prefers-color-scheme: dark)').matches
  var effective = preference === 'system' ? (systemPrefersDark ? 'dark' : 'light') : preference
  document.documentElement.classList.toggle('dark', effective === 'dark')
})()
