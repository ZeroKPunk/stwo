use crate::core::air::accumulation::PointEvaluationAccumulator;
use crate::core::air::Component;
use crate::core::circle::CirclePoint;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::TreeVec;
use crate::core::{ColumnVec, InteractionElements};

pub const XOR_Z_ID: &str = "xor_z";

pub const XOR_ALPHA_ID: &str = "xor_alpha";

/// 8-bit XOR lookup table.
pub struct XorTableComponent;

impl Component for XorTableComponent {
    fn n_constraints(&self) -> usize {
        0
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        u8::BITS + u8::BITS
    }

    fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>> {
        TreeVec::new(vec![vec![u8::BITS + u8::BITS]])
    }

    fn mask_points(
        &self,
        _point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        TreeVec::new(vec![])
    }

    fn interaction_element_ids(&self) -> Vec<String> {
        vec![XOR_ALPHA_ID.to_string(), XOR_Z_ID.to_string()]
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        _point: CirclePoint<SecureField>,
        _mask: &ColumnVec<Vec<SecureField>>,
        _evaluation_accumulator: &mut PointEvaluationAccumulator,
        _interaction_elements: &InteractionElements,
    ) {
        todo!()
    }
}
