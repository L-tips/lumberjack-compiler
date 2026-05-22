use core::fmt;
use half::bf16;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, byteorder::little_endian::U16};

/// A specialized relative pointer for use with optimized trees.
///
/// It can be interpreted either as a relative pointer or a [`bf16`], depending on context.
#[repr(transparent)]
#[derive(Clone, Copy, IntoBytes, KnownLayout, Immutable, FromBytes)]
pub struct NodePointer(U16);

impl NodePointer {
    pub fn new_ptr(ptr: u16) -> Self {
        Self(U16::new(ptr))
    }

    pub fn new_bf16(float: bf16) -> Self {
        // let float = F32::new(float);
        Self(U16::from_bytes(float.to_le_bytes()))
    }

    /// Return the pointer representation as a raw integer.
    pub fn as_ptr(&self) -> u16 {
        self.0.get()
    }

    pub fn as_bf16(&self) -> bf16 {
        let bytes = self.0.to_bytes();
        bf16::from_le_bytes(bytes)
    }
}

impl fmt::Debug for NodePointer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NodePointer: {{ bytes: {:?}, (as_u32: {}, as_f32: {}) }}",
            self.0.as_bytes(),
            self.as_ptr(),
            self.as_bf16()
        )
    }
}

impl fmt::Display for NodePointer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NodePointer: {:?} (u32: {}, f32: {})",
            self.0.as_bytes(),
            self.as_ptr(),
            self.as_bf16()
        )
    }
}
