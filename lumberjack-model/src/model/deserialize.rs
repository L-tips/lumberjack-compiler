use core::num::NonZeroU16;

use zerocopy::little_endian::U16;

use crate::{
    Error,
    model::{ALIGNMENT, Branch, Model, Node, u24},
};

impl<'a> Model<'a> {
    pub fn deserialize(buffer: &'a [u8]) -> Result<Self, Error> {
        let base_ptr = buffer.as_ptr();

        // Ensure alignment
        assert_eq!(base_ptr as usize % align_of::<Self>(), 0);

        // Ensure we have enough data for the fixed-size part of ConcreteType
        let header_len = size_of::<u24>()  // num_trees
            + size_of::<u8>()                  // num_cells
            + size_of::<u16>()                 // num_features
            + size_of::<u16>()                 // num_targets
            + size_of::<u64>(); // padding

        // Ensure we at least have enough data for all fields + at least 1 node
        assert!(buffer.len() >= header_len + size_of::<Node>());

        unsafe {
            // Number of trees (3 bytes)
            let ptr = base_ptr as *const u24;
            let num_trees = *ptr;

            let ptr = ptr.add(1) as *const u8;
            let num_cells = *ptr;

            // Number of features (2 bytes)
            let ptr = ptr.add(1) as *const U16;
            let num_features = *ptr;
            NonZeroU16::new(num_features.get()).ok_or(Error::NoFeatures)?;

            // Number of targets (2 bytes)
            let c_ptr = ptr.add(1);
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

            Ok(Model {
                num_trees,
                num_cells,
                num_features,
                num_targets,
                _padding: 0,
                nodes,
            })
        }
    }
}
