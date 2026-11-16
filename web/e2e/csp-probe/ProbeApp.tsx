import * as Dialog from '@radix-ui/react-dialog'
import * as Select from '@radix-ui/react-select'

/**
 * Permanent regression fixture for the T08 finding (design §6.3): Radix's
 * `react-remove-scroll` injects a `<style>` tag for scrollbar-width
 * compensation, and `Select.Viewport` injects its own inline `<style>`.
 * Under `style-src 'self'` alone this is blocked; the fix is a per-request
 * style nonce (see ./vite.config.ts `cspNonceProbePlugin` and ./main.tsx).
 * Trimmed to the two affected primitives -- not the full app -- since this
 * fixture only needs to prove the nonce mechanism still works.
 */
export function ProbeApp({ nonce }: { nonce: string | undefined }) {
  return (
    <div className="scroll-filler">
      <Dialog.Root>
        <Dialog.Trigger data-testid="trigger-dialog">Open dialog</Dialog.Trigger>
        <Dialog.Portal>
          <Dialog.Overlay />
          <Dialog.Content>
            <Dialog.Title>Probe dialog</Dialog.Title>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>

      <Select.Root defaultValue="a">
        <Select.Trigger data-testid="trigger-select">
          <Select.Value />
        </Select.Trigger>
        <Select.Portal>
          <Select.Content>
            {/* design §6.3: Select.Viewport needs the nonce prop explicitly,
                separately from react-remove-scroll's get-nonce-based injection. */}
            <Select.Viewport nonce={nonce}>
              <Select.Item value="a">
                <Select.ItemText>A</Select.ItemText>
              </Select.Item>
              <Select.Item value="b">
                <Select.ItemText>B</Select.ItemText>
              </Select.Item>
            </Select.Viewport>
          </Select.Content>
        </Select.Portal>
      </Select.Root>
    </div>
  )
}
