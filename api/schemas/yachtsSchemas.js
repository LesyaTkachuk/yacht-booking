import Joi from "joi";

export const createYachtSchema = Joi.object({
  name: Joi.string().min(2).max(100).required(),
  type: Joi.string().min(2).max(100).required(),
  guests: Joi.number().min(1).max(10000).required(),
  cabins: Joi.number().min(1).max(1000).required(),
  crew: Joi.number().min(1).max(1000).optional(),
  length: Joi.number().min(1).max(1000).required(),
  year: Joi.number().min(1900).max(2025).required(),
  model: Joi.string().min(2).max(100).optional(),
  summerLowSeasonPrice: Joi.number().min(1).max(500000).required(),
  summerHighSeasonPrice: Joi.number().min(1).max(500000).required(),
  winterLowSeasonPrice: Joi.number().min(1).max(500000).required(),
  winterHighSeasonPrice: Joi.number().min(1).max(500000).required(),
  rating: Joi.number().min(0).max(5).optional(),
  country: Joi.string().min(2).max(100).required(),
  baseMarina: Joi.string().min(2).max(100).optional(),
  description: Joi.string().min(2).max(1000).optional(),
  photos: Joi.array().items(Joi.string()).optional(),
});

export const updateYachtSchema = Joi.object({
  name: Joi.string().min(2).max(100),
  type: Joi.string().min(2).max(100),
  guests: Joi.number().min(1).max(10000),
  cabins: Joi.number().min(1).max(1000),
  crew: Joi.number().min(1).max(1000),
  length: Joi.number().min(1).max(1000),
  year: Joi.number().min(1900).max(2025),
  model: Joi.string().min(2).max(100).optional(),
  summerLowSeasonPrice: Joi.number().min(1).max(10000),
  summerHighSeasonPrice: Joi.number().min(1).max(10000),
  winterLowSeasonPrice: Joi.number().min(1).max(10000),
  winterHighSeasonPrice: Joi.number().min(1).max(10000),
  model: Joi.string().min(2).max(1000),
  rating: Joi.number().min(0).max(5),
  baseMarina: Joi.string().min(2).max(100),
  description: Joi.string().min(2).max(1000),
})
  .min(1)
  .message("Body must have at least one field");

export const updateYachtRatingSchema = Joi.object({
  rating: Joi.boolean().required(),
});
