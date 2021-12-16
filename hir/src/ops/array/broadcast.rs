use crate::infer::*;
use crate::internal::*;

use tract_core::ops::array::MultiBroadcastTo as Typed;

#[derive(Debug, Clone, new, Default, Hash)]
pub struct MultiBroadcastTo;
impl_dyn_hash!(MultiBroadcastTo);

impl Op for MultiBroadcastTo {
    fn name(&self) -> Cow<str> {
        "MultiBroadcastTo".into()
    }

    op_hir!();
    op_as_typed_op!();
}

impl EvalOp for MultiBroadcastTo {
    fn is_stateless(&self) -> bool {
        false
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (data, shape) = args_2!(inputs);
        let shape = shape.cast_to::<TDim>()?;
        let dims: Vec<usize> = shape
            .as_slice::<TDim>()?
            .iter()
            .map(|d| Ok(d.to_usize()?))
            .collect::<TractResult<_>>()?;
        dispatch_datum_by_size!(Typed::eval_t(data.datum_type())(&*data, &*dims))
    }
}

impl TypedOp for MultiBroadcastTo {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let rank = inputs[0].rank().max(inputs[1].shape.volume().to_usize()?);
        let one = 1.to_dim();
        let spec: TVec<TDim> = inputs[1]
            .konst
            .as_ref()
            .map(|k| k.cast_to::<TDim>().map(|k| k.to_owned().as_slice::<TDim>().unwrap().into()))
            .transpose()?
            .unwrap_or_else(|| ('a'..).take(rank).map(|c| Symbol::new(c).into()).collect());
        let output_shape: TVec<TDim> = (0..rank)
            .map(|axis| {
                let left = inputs[0].shape.get(rank - inputs[0].rank() + axis).unwrap_or(&one);
                let right = spec.get(rank - spec.len() + axis).unwrap_or(&one);
                let d = if left.is_one() {
                    right
                } else if right.is_one() {
                    left
                } else if left == right {
                    left
                } else if let (Ok(l), Ok(r)) = (left.to_usize(), right.to_usize()) {
                    bail!("Can not broadcast {} and {}", l, r)
                } else {
                    right
                };
                Ok(d.clone())
            })
            .collect::<TractResult<TVec<TDim>>>()?;
        Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, output_shape)))
    }

    as_op!();
}

impl InferenceRulesOp for MultiBroadcastTo {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 2)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&inputs[1].rank, 1)?;
        s.given_2(&inputs[1].shape[0], &inputs[0].rank, move |s, l, r| {
            if let Ok(l) = l.to_i64() {
                let output_rank = l.max(r);
                s.equals(&outputs[0].rank, output_rank)?;
            }
            Ok(())
        })?;
        s.given(&inputs[0].shape, move |s, shape| {
            s.given(&inputs[1].value, move |s, dims| {
                let dims = dims.cast_to::<TDim>()?;
                let dims =
                    tract_core::broadcast::multi_broadcast(&[&*dims.as_slice::<TDim>()?, &*shape])
                        .with_context(|| format!("broadcasting {:?} to {:?}", shape, dims))?;
                s.equals(&outputs[0].shape, ShapeFactoid::from(dims))
            })
        })
    }

    as_op!();
    to_typed!();
}

/*
fn wire(
&self,
prefix: &str,
model: &mut TypedModel,
inputs: &[OutletId],
) -> TractResult<TVec<OutletId>> {
if let Some(shape) = model.outlet_fact(inputs[1])?.konst.clone() {
let input_shape = model.outlet_fact(inputs[0])?.shape.to_tvec();
let shape = shape.cast_to::<TDim>()?;
let shape = shape.as_slice::<TDim>()?;
let dims = tract_core::broadcast::multi_broadcast(&[&*input_shape, &*shape])
.with_context(|| format!("broadcasting {:?} to {:?}", input_shape, shape))?;
let op = Typed::new(dims.into());
model.wire_node(prefix, op, &[inputs[0]])
} else {
bail!("shape input is variable")
}
}
}
*/
