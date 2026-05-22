use core::{marker::PhantomData, num::NonZeroU8, ops::Deref};

use zerocopy::byteorder::little_endian::U32;

use crate::Error;

use super::{Branch, OptimizedForest, ProblemType};

#[macro_export]
macro_rules! static_storage {
    ($file:expr $(, unsafe(link_section = $section:literal))?) => {{
        const BYTES_LEN: usize = include_bytes!($file).len();

        $(#[unsafe(link_section = $section)])?
        static BUF: ::embedded_rforest::forest::deserialize::BackingStorage<BYTES_LEN> =
            ::embedded_rforest::forest::deserialize::BackingStorage::new(*include_bytes!($file));
        BUF.to_slice()
    }};
}

#[cfg_attr(
    any(target_pointer_width = "32", target_pointer_width = "16"),
    repr(align(4))
)]
#[cfg_attr(target_pointer_width = "64", repr(align(8)))]
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

impl<'a, P: ProblemType> OptimizedForest<'a, P> {
    pub fn deserialize(buffer: &'a [u8]) -> Result<Self, Error> {
        let base_ptr = buffer.as_ptr();

        // Ensure alignment
        assert_eq!(base_ptr as usize % align_of::<Self>(), 0);

        // Ensure we have enough data for the fixed-size part of ConcreteType
        let header_size = size_of::<u32>()  // num_trees
            + size_of::<u8>()               // num_features
            + size_of::<u8>()               // num_targets
            + 2                             // padding
            + size_of::<Branch>(); // At least 1 node

        // Ensure we at least have enough data for all fields
        assert!(buffer.len() >= header_size);

        unsafe {
            // Number of trees (4 bytes)
            let a_ptr = base_ptr as *const u32;
            let num_trees = U32::new(*a_ptr);

            // Number of features (1 byte)
            let b_ptr = a_ptr.add(1) as *const u8;
            let num_features = *b_ptr;

            // Number of targets (1 byte)
            let c_ptr = b_ptr.add(1);
            let num_targets = if *c_ptr == 0 {
                None
            } else {
                Some(NonZeroU8::new_unchecked(*c_ptr))
            };

            // Check that the forest is of the correct problem type according to the P type
            // parameter
            if (num_targets.is_some() && !P::HAS_TARGETS)
                || (num_targets.is_none() && P::HAS_TARGETS)
            {
                return Err(Error::WrongProblemType);
            }

            // Get start of node slice and skip padding (2 bytes)
            let header_len = size_of::<u32>() + size_of::<u8>() * 2 + 2;
            let slice_size = buffer.len() - header_len;

            if !slice_size.is_multiple_of(size_of::<Branch>()) {
                return Err(Error::MalformedForest);
            }

            let slice_len = slice_size / size_of::<Branch>();
            let slice_ptr = (base_ptr.byte_add(header_len)) as *const Branch;
            let branch_slice = core::slice::from_raw_parts(slice_ptr, slice_len);

            for branch in branch_slice.iter() {
                if !branch.flags.left_prediction() && (branch.left.as_ptr() as usize) >= slice_len {
                    return Err(Error::MalformedForest);
                }
                if !branch.flags.right_prediction() && (branch.right.as_ptr() as usize) >= slice_len
                {
                    return Err(Error::MalformedForest);
                };
            }

            Ok(OptimizedForest {
                num_trees,
                num_features,
                num_targets,
                _padding: [0; 2],
                nodes: branch_slice,
                _problem: PhantomData,
            })
        }
    }
}
