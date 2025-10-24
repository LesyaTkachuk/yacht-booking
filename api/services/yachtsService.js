import Yacht from "../db/models/Yacht.js";
import { DEFAULT_PAGE, DEFAULT_LIMIT } from "../constants/yachts.js";

// TODO change all endpoints
export const listYachts = async (query) => {
  const {
    page: queryPage = DEFAULT_PAGE,
    limit: queryLimit = DEFAULT_LIMIT,
    ...restQuery
  } = query;

  // check for numeric values
  const page = Math.max(Number(queryPage), 1);
  const limit = Math.max(Number(queryLimit), 1);

  return await Yacht.findAll({
    where: restQuery,
    limit,
    offset: (page - 1) * limit,
  });
};

export const getYachtById = (yachtId) => Yacht.findByPk(yachtId);

export const getYacht = (query) => Yacht.findOne({ where: query });

export const removeYacht = (query) => Yacht.destroy({ where: query });

export const addYacht = (data) => Yacht.create(data);

export const updateYacht = async (query, data) => {
  const yacht = await getYacht(query);

  if (!yacht) {
    return null;
  }

  return await yacht.update(data, {
    returning: true,
  });
};

export const updateYachtRating = async (query, rating) => {
  const yacht = await getYacht(query);
  if (!yacht) {
    return null;
  }
  return yacht.update({ rating }, { returning: true });
};
