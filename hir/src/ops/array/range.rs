use crate::internal::*;
use tract_ndarray::prelude::*;

#[derive(Debug, Default, Clone, new, Hash)]
pub struct Range;

impl_dyn_hash!(Range);

impl Op for Range {
    fn name(&self) -> Cow<str> {
        "Range".into()
    }

    op_hir!();
    op_as_typed_op!();
}

impl TypedOp for Range {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, &[TDim::from(Symbol::new('r'))])))
    }

    as_op!();
}

impl EvalOp for Range {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (start, limit, steps) = args_3!(inputs);
        let dt = start.datum_type();
        let start = start.cast_to_scalar::<u64>()?;
        let limit = limit.cast_to_scalar::<u64>()?;
        let steps = steps.cast_to_scalar::<u64>()?;
        let value = Array1::from_shape_fn(((limit - start) / steps) as usize, |ix| {
            ix as u64 * steps + start
        });
        let value = value.into_tensor().cast_to_dt(dt)?.into_owned();
        Ok(tvec!(value.into_arc_tensor()))
    }
}

impl InferenceRulesOp for Range {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 3)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &inputs[1].datum_type)?;
        s.equals(&inputs[0].datum_type, &inputs[2].datum_type)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, 0)?;
        s.equals(&inputs[1].rank, 0)?;
        s.equals(&inputs[2].rank, 0)?;
        s.equals(&outputs[0].rank, 1)?;
        Ok(())
    }

    to_typed!();
    as_op!();
}
