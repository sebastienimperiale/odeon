//! The observers viewer as a client of an `observers-server`: the same
//! interface as `observers-viewer`, but every estimator job is posted to
//! the server as a *twin experiment* (`POST /twin-runs`,
//! `ode_observers_remote::RemoteRun::spawn_twin`: a few kilobytes — the
//! server regenerates the reference from the same seeds) and its output
//! fetched back; only the trajectories on screen are computed here. Builds as a desktop
//! application and as a web page (wasm, `trunk`).
//!
//! Desktop: the server URL comes from `ODEON_SERVER` (default
//! `http://127.0.0.1:8787`) and the server's token, if it needs one, from
//! `ODEON_TOKEN`; both are editable in the estimator section.
//!
//! ```sh
//! cargo run -p odeon-observers-server --release      # terminal 1
//! cargo run -p odeon-observers-client --release      # terminal 2
//! ```
//!
//! Web: the URL is, in order, the `?server=https://…` query parameter of
//! the page's address (remembered in the browser's local storage), the
//! remembered value, or the page's own directory (its origin for a page
//! at `/`; `https://host/odeon` for a page at `/odeon/`, the server
//! sitting behind a reverse proxy that strips the prefix) — so a page
//! served by the server itself needs no configuration, and a page published elsewhere
//! (GitHub Pages, say) is shared as one link naming the server:
//! `https://you.github.io/Odeon/?server=https://my-mac.example.net`.
//! Editing the field in the estimator section updates the remembered
//! value. The token works the same way: `?token=…`, remembered under
//! `odeon.token`, editable. (A page served over HTTPS can only call an
//! HTTPS server.)
//!
//! ```sh
//! cd apps/observers-client && trunk build --release        # → dist/
//! ODEON_WEB_DIR=apps/observers-client/dist cargo run -p odeon-observers-server --release
//! # then open http://127.0.0.1:8787
//! ```
//!
//! (`trunk serve` in `apps/observers-client` also works for development;
//! the page then talks to the server cross-origin, which it allows.)

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use eframe::egui;
use ode_models_spec::spec::TwinSpec;
use ode_observers::jobs::{JobOutput, JobSpec};
use ode_observers_egui::app::{App, Compute, EstimatorRun, LocalRunner};
use ode_observers_remote::protocol::TwinJob;
use ode_observers_remote::RemoteRun;

/// Jobs posted to the observers server at `url`, as twin experiments,
/// with its bearer `token` when it requires one (empty = none).
struct RemoteCompute {
    url: String,
    token: String,
}

impl Default for RemoteCompute {
    /// Desktop: `ODEON_SERVER` or the local default port, `ODEON_TOKEN`.
    /// Web: the `?server=` / `?token=` parameters, the remembered values,
    /// or the page's origin and no token.
    fn default() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        let (url, token) = {
            let var = |name: &str| std::env::var(name).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
            (var("ODEON_SERVER").unwrap_or_else(|| "http://127.0.0.1:8787".to_string()), var("ODEON_TOKEN").unwrap_or_default())
        };
        #[cfg(target_arch = "wasm32")]
        let (url, token) = (web::server_url(), web::token());
        RemoteCompute { url, token }
    }
}

/// The directory a page lives in, as a URL without its trailing slash:
/// the origin plus the path up to its last `/` (`/` → the origin,
/// `/odeon/` and `/odeon/index.html` → `<origin>/odeon`).
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn page_directory(origin: &str, pathname: &str) -> String {
    let dir = pathname.rfind('/').map_or("", |end| &pathname[..end]);
    format!("{}{dir}", origin.trim_end_matches('/'))
}

/// Where the web page learns the server URL and token from, and remembers
/// them.
#[cfg(target_arch = "wasm32")]
mod web {
    pub const SERVER_KEY: &str = "odeon.server";
    pub const TOKEN_KEY: &str = "odeon.token";

    fn storage() -> Option<web_sys::Storage> {
        web_sys::window()?.local_storage().ok().flatten()
    }

    /// Remember `value` under `key` for the next visit (best effort).
    pub fn remember(key: &str, value: &str) {
        if let Some(storage) = storage() {
            let _ = storage.set_item(key, value.trim());
        }
    }

    /// `?name=…` of the page's address, if present and non-empty.
    fn query(name: &str) -> Option<String> {
        web_sys::window()?
            .location()
            .search()
            .ok()
            .and_then(|q| web_sys::UrlSearchParams::new_with_str(&q).ok())
            .and_then(|p| p.get(name))
            .map(|u| u.trim().to_string())
            .filter(|u| !u.is_empty())
    }

    fn remembered(key: &str) -> Option<String> {
        storage()?.get_item(key).ok().flatten().map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
    }

    /// The query parameter (remembered when present), else the remembered
    /// value.
    fn setting(name: &str, key: &str) -> Option<String> {
        if let Some(value) = query(name) {
            remember(key, &value);
            return Some(value);
        }
        remembered(key)
    }

    /// `?server=…`, else the remembered URL, else the page's own directory
    /// — its origin for a page at `/`, `https://host/odeon` for a page at
    /// `/odeon/` behind a reverse proxy stripping that prefix.
    pub fn server_url() -> String {
        setting("server", SERVER_KEY).unwrap_or_else(|| {
            web_sys::window()
                .and_then(|w| {
                    let location = w.location();
                    Some(super::page_directory(&location.origin().ok()?, &location.pathname().ok()?))
                })
                .unwrap_or_else(|| "http://127.0.0.1:8787".to_string())
        })
    }

    /// `?token=…`, else the remembered token, else none.
    pub fn token() -> String {
        setting("token", TOKEN_KEY).unwrap_or_default()
    }
}

/// A run on the server, seen through the app's run trait.
struct Remote<T>(RemoteRun<T>);

impl<T: TryFrom<JobOutput, Error = String>> EstimatorRun<T> for Remote<T> {
    fn progress(&self) -> f32 {
        self.0.progress()
    }
    fn elapsed(&self) -> f64 {
        self.0.elapsed()
    }
    fn try_take(&self) -> Option<T> {
        self.0.try_take()
    }
    fn error(&self) -> Option<String> {
        self.0.error()
    }
}

impl Compute for RemoteCompute {
    fn spawn<T>(&self, job: JobSpec, twin: TwinSpec, _local: LocalRunner<T>) -> Box<dyn EstimatorRun<T>>
    where
        T: TryFrom<JobOutput, Error = String> + Send + 'static,
    {
        let JobSpec { estimator, config, .. } = job;
        let token = Some(self.token.as_str()).filter(|t| !t.trim().is_empty());
        Box::new(Remote(RemoteRun::spawn_twin_with_token(&self.url, token, &TwinJob { twin, estimator, config })))
    }

    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("server").on_hover_text(
                "URL of the observers-server every estimator job is sent to (start one \
                 with `cargo run -p odeon-observers-server --release`; ODEON_SERVER in \
                 the environment pre-fills this field).",
            );
            let edited = ui
                .add(egui::TextEdit::singleline(&mut self.url).desired_width(ui.available_width() - 8.0))
                .changed();
            #[cfg(target_arch = "wasm32")]
            if edited {
                web::remember(web::SERVER_KEY, &self.url);
            }
            #[cfg(not(target_arch = "wasm32"))]
            let _ = edited;
        });
        ui.horizontal(|ui| {
            ui.label("token").on_hover_text(
                "Access token of the server, if it was started with ODEON_TOKEN (leave \
                 empty for an open server; ODEON_TOKEN in the environment pre-fills it).",
            );
            let edited = ui
                .add(egui::TextEdit::singleline(&mut self.token).password(true).desired_width(ui.available_width() - 8.0))
                .changed();
            #[cfg(target_arch = "wasm32")]
            if edited {
                web::remember(web::TOKEN_KEY, &self.token);
            }
            #[cfg(not(target_arch = "wasm32"))]
            let _ = edited;
        });
    }
}

struct Client(App<RemoteCompute>);

impl eframe::App for Client {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.0.ui(ui);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1150.0, 780.0])
            .with_title("Odeon — observers (server)"),
        ..Default::default()
    };
    eframe::run_native(
        "Odeon observers client",
        options,
        Box::new(|_cc| Ok(Box::new(Client(App::new(RemoteCompute::default()))))),
    )
}

/// The web page: the app on the `<canvas id="odeon">` of `index.html`.
#[cfg(target_arch = "wasm32")]
fn main() {
    eframe::WebLogger::init(log::LevelFilter::Debug).ok();
    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window().and_then(|w| w.document()).expect("no document");
        let canvas = document
            .get_element_by_id("odeon")
            .expect("no <canvas id=\"odeon\"> in index.html")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("#odeon is not a canvas");
        eframe::WebRunner::new()
            .start(
                canvas,
                eframe::WebOptions::default(),
                Box::new(|_cc| Ok(Box::new(Client(App::new(RemoteCompute::default()))))),
            )
            .await
            .expect("failed to start the web app");
    });
}

#[cfg(target_arch = "wasm32")]
use eframe::wasm_bindgen::JsCast as _;

#[cfg(test)]
mod tests {
    use super::page_directory;

    #[test]
    fn the_default_server_is_the_directory_of_the_page() {
        let origin = "https://team-ananke.fr";
        assert_eq!(page_directory(origin, "/"), origin);
        assert_eq!(page_directory(origin, "/index.html"), origin);
        assert_eq!(page_directory(origin, "/odeon/"), "https://team-ananke.fr/odeon");
        assert_eq!(page_directory(origin, "/odeon/index.html"), "https://team-ananke.fr/odeon");
        assert_eq!(page_directory(origin, ""), origin);
    }
}
