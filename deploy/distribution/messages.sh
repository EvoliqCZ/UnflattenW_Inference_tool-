#!/usr/bin/env bash
# Shared terminal copy. Browser language is selected independently in the UI.
case "${UNFLATTEN_APP_LANG:-${LC_ALL:-${LC_MESSAGES:-${LANG:-en}}}}" in
    sk|sk_*|sk-*) UNFLATTEN_LAUNCH_LANGUAGE=sk ;;
    *) UNFLATTEN_LAUNCH_LANGUAGE=en ;;
esac
message() {
    local english slovak
    case "$1" in
        unsupported_path)
            english='The path contains unsupported characters. Use a regular folder; spaces are allowed.'
            slovak='Cesta obsahuje nepodporované znaky. Použite bežný priečinok; medzery sú povolené.' ;;
        invalid_port) english='Invalid port.'; slovak='Neplatný port.' ;;
        invalid_saved_port) english='Invalid saved port.'; slovak='Neplatný uložený port.' ;;
        missing_port) english='The saved configuration has no port.'; slovak='V uloženej konfigurácii chýba port.' ;;
        invalid_hf_token) english='Invalid Hugging Face token format.'; slovak='Neplatný formát Hugging Face tokenu.' ;;
        local_data) english='Local data: '; slovak='Lokálne dáta: ' ;;
        docker_consent_required)
            english='The PDF sandbox needs the local Docker socket. After reviewing this access, enable UNFLATTEN_ACCEPT_ROOTFUL_SOCKET=1.'
            slovak='PDF sandbox potrebuje lokálny Docker socket. Po oboznámení sa s prístupom povoľte UNFLATTEN_ACCEPT_ROOTFUL_SOCKET=1.' ;;
        docker_consent_prompt)
            english='The PDF sandbox uses the Docker socket. The worker can therefore control Docker on this computer.
Allow this mode on your trusted computer? [yes/N]: '
            slovak='PDF sandbox používa Docker socket. Worker tým môže ovládať Docker na tomto počítači.
Povoliť tento režim na vašom dôveryhodnom počítači? [ano/N]: ' ;;
        docker_consent_denied) english='Docker socket access was not allowed.'; slovak='Prístup k Docker socketu nebol povolený.' ;;
        gpu_prepare) english='The model will be prepared on the GPU automatically.'; slovak='Model sa automaticky pripraví v GPU.' ;;
        install_docker) english='Install Docker Engine, Compose and NVIDIA Container Toolkit.'; slovak='Nainštalujte Docker Engine, Compose a NVIDIA Container Toolkit.' ;;
        docker_unavailable) english='Docker is not available. Check that it is running and your account has access.'; slovak='Docker nie je dostupný. Skontrolujte jeho spustenie a oprávnenia.' ;;
        missing_compose) english='The Docker Compose plugin is missing.'; slovak='Chýba Docker Compose plugin.' ;;
        gpu_unavailable) english='An NVIDIA GPU is not available. This version requires a CUDA GPU; there is no CPU mode.'; slovak='NVIDIA GPU nie je dostupná. Táto verzia potrebuje CUDA GPU; CPU režim sa nepoužije.' ;;
        preparing) english='Preparing the application. The first build and model download may take a while.'; slovak='Pripravujem aplikáciu. Prvé zostavenie a stiahnutie modelu môže trvať dlhšie.' ;;
        sandbox_unverified) english='Cannot verify the built PDF sandbox image.'; slovak='Nie je možné overiť zostavený obraz PDF sandboxu.' ;;
        not_configured) english='The application has not been configured yet.'; slovak='Aplikácia ešte nebola nastavená.' ;;
        usage) english='Usage: ./Start.sh [start|restart|configure|status|logs|stop|reset-password]'; slovak='Použitie: ./Spustit.sh [start|restart|configure|status|logs|stop|reset-password]' ;;
        shortcut_ready) english='The Unflatten W 1.0 launcher is ready in the applications menu.'; slovak='Spúšťač Unflatten W 1.0 je pripravený v ponuke aplikácií.' ;;
        *) printf 'Unknown launcher message: %s' "$1" >&2; return 1 ;;
    esac
    if [[ "$UNFLATTEN_LAUNCH_LANGUAGE" == sk ]]; then printf '%s' "$slovak"; else printf '%s' "$english"; fi
}
