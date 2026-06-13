use std::{os::raw::c_void, ptr::NonNull};

use objc::runtime::Object;
use raw_window_handle as rwh;
#[derive(Debug, Clone, Copy)]
pub(crate) struct RawWindow {
    pub(crate) handle: *mut Object,
}

// Safety: The raw pointer in RawWindow point to MacOS window
// which is valid for the window's lifetime. This is used only for
// passing to wgpu which needs Send+Sync for surface creation.
unsafe impl Send for RawWindow {}
unsafe impl Sync for RawWindow {}

impl rwh::HasWindowHandle for RawWindow {
    fn window_handle(&self) -> Result<rwh::WindowHandle<'_>, rwh::HandleError> {
        let window =
            rwh::AppKitWindowHandle::new(NonNull::new(self.handle as *mut c_void).unwrap());
        Ok(unsafe { rwh::WindowHandle::borrow_raw(window.into()) })
    }
}
impl rwh::HasDisplayHandle for RawWindow {
    fn display_handle(&self) -> Result<rwh::DisplayHandle<'_>, rwh::HandleError> {
        Ok(unsafe {
            rwh::DisplayHandle::borrow_raw(rwh::RawDisplayHandle::AppKit(
                rwh::AppKitDisplayHandle::new(),
            ))
        })
    }
}
