use core::ops::Deref;

#[macro_export]
macro_rules! static_storage {
    ($file:expr $(, unsafe(link_section = $section:literal))?) => {{
        const BYTES_LEN: usize = include_bytes!($file).len();

        $(#[unsafe(link_section = $section)])?
        static BUF: ::lumberjack_model::BackingStorage<BYTES_LEN> =
            ::lumberjack_model::BackingStorage::new(*include_bytes!($file));
        BUF.to_slice()
    }};
}

#[repr(align(16))]
pub struct BackingStorage<const N: usize>([u8; N]);

impl<const N: usize> BackingStorage<N> {
    pub const fn new(buf: [u8; N]) -> Self {
        Self(buf)
    }

    pub const fn to_slice(&self) -> &[u8] {
        self.0.as_slice()
    }
}

impl<const N: usize> Deref for BackingStorage<N> {
    type Target = [u8; N];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
