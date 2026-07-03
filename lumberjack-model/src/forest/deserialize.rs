use core::num::NonZeroU16;
use core::ops::Deref;

use zerocopy::{byteorder::little_endian::U32, little_endian::U16};

use crate::{
    Error,
    forest::{ALIGNMENT, Node},
};

use super::{Branch, OptimizedForest};

#[macro_export]
macro_rules! static_storage {
    ($file:expr $(, unsafe(link_section = $section:literal))?) => {{
        const BYTES_LEN: usize = include_bytes!($file).len();

        $(#[unsafe(link_section = $section)])?
        static BUF: ::lumberjack_model::forest::deserialize::BackingStorage<BYTES_LEN> =
            ::lumberjack_model::forest::deserialize::BackingStorage::new(*include_bytes!($file));
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

impl<'a> OptimizedForest<'a> {
    pub fn deserialize(buffer: &'a [u8]) -> Result<Self, Error> {
        let base_ptr = buffer.as_ptr();

        // Ensure alignment
        assert_eq!(base_ptr as usize % align_of::<Self>(), 0);

        // Ensure we have enough data for the fixed-size part of ConcreteType
        let header_len = size_of::<u32>()  // num_trees
            + size_of::<u16>()               // num_features
            + size_of::<u16>()               // num_targets
            + size_of::<u64>(); // padding

        // Ensure we at least have enough data for all fields + at least 1 node
        assert!(buffer.len() >= header_len + size_of::<Node>());

        unsafe {
            // Number of trees (4 bytes)
            let a_ptr = base_ptr as *const u32;
            let num_trees = U32::new(*a_ptr);

            // Number of features (2 bytes)
            let b_ptr = a_ptr.add(1) as *const U16;
            let num_features = *b_ptr;
            NonZeroU16::new(num_features.get()).ok_or(Error::NoFeatures)?;

            // Number of targets (2 bytes)
            let c_ptr = b_ptr.add(1);
            let num_targets = *c_ptr;
            NonZeroU16::new(num_targets.get()).ok_or(Error::NoTargets)?;

            // Get start of node slice and skip padding (8 bytes)
            let slice_size = buffer.len() - header_len;

            if !slice_size.is_multiple_of(size_of::<Branch>()) {
                return Err(Error::MalformedForest);
            }

            let slice_len = slice_size / size_of::<Node>();
            let slice_ptr = (base_ptr.byte_add(header_len)) as *const Node;

            // Node list must be 128-bit aligned
            if !(slice_ptr as usize).is_multiple_of(ALIGNMENT) {
                return Err(Error::MisalignedData);
            }

            let nodes = core::slice::from_raw_parts(slice_ptr, slice_len);

            Ok(OptimizedForest {
                num_trees,
                num_features,
                num_targets,
                _padding: 0,
                nodes,
            })
        }
    }
}
