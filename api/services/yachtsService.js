import { spawn } from "child_process";
import path from "path";
import { fileURLToPath } from "url";
import Yacht from "../db/models/Yacht.js";
import User from "../db/models/User.js";
import Event from "../db/models/Event.js";
import { DEFAULT_PAGE, DEFAULT_LIMIT } from "../constants/yachts.js";
import { Op } from "sequelize";
import dotenv from "dotenv";

const getBookedYachtIds = async () => {
  // collect yachtIds that have been booked at least once
  const bookedRows = await Event.findAll({
    attributes: ["yachtId"],
    where: { type: "book" },
    group: ["yachtId"],
    raw: true,
  });

  const bookedIds = bookedRows.map((r) => r.yachtId);
  if (bookedIds.length === 0) {
    return [];
  }

  return bookedIds;
};
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

export const addYacht = async (data) => {
  const newYacht = await Yacht.create(data);

  triggerSimilarYachtsUpdate();

  return newYacht;
};

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

export const getTopBookedYachts = async () => {
  const bookedIds = await getBookedYachtIds();

  return await Yacht.findAll({
    where: {
      id: { [Op.in]: bookedIds },
    },
    order: [["rating", "DESC"]],
    limit: 12,
  });
};

export const getRecommendations = async (query) => {
  const user = await User.findOne({ where: query });

  if (!user) {
    return next(HttpError(401, USER_NOT_FOUND));
  }

  const ids = Array.isArray(user.recommendations) ? user.recommendations : [];

  // --- Cold start path ---
  if (ids.length === 0) {
    const bookedIds = await getBookedYachtIds();

    // build budget filter: [budgetMin, budgetMax * 1.1]
    const hasBudgetMin = Number.isFinite(user.budgetMin);
    const hasBudgetMax = Number.isFinite(user.budgetMax);
    const minPrice = hasBudgetMin ? user.budgetMin : 0;
    const maxPrice = hasBudgetMax ? Math.floor(user.budgetMax * 1.1) : null;

    const priceWhere =
      maxPrice != null
        ? { summerLowSeasonPrice: { [Op.between]: [minPrice, maxPrice] } }
        : { summerLowSeasonPrice: { [Op.gte]: minPrice } };

    // query booked yachts within budget; take the 12 highest priced
    const topByPrice = await Yacht.findAll({
      where: {
        id: { [Op.in]: bookedIds },
        ...(hasBudgetMin || hasBudgetMax ? priceWhere : {}),
      },
      order: [["summerLowSeasonPrice", "DESC"]],
      limit: 12,
    });

    // If we couldn't fill 12 due to the budget filter, backfill with remaining booked yachts (still by price)
    if (topByPrice.length < 12) {
      const excludeIds = topByPrice.map((y) => y.id);
      const backfill = await Yacht.findAll({
        where: {
          id: {
            [Op.in]: bookedIds.filter((id) => !excludeIds.includes(id)),
          },
        },
        order: [["summerLowSeasonPrice", "DESC"]],
        limit: 12 - topByPrice.length,
      });

      topByPrice.push(...backfill);
    }

    // sort by rating
    const coldStart = topByPrice
      .sort((a, b) => {
        const ra = Number.isFinite(a.rating) ? a.rating : 0;
        const rb = Number.isFinite(b.rating) ? b.rating : 0;
        return rb - ra;
      })
      .slice(0, 12);

    return coldStart;
  }

  // --- Personalized path ---
  const personalRecommendations = await Yacht.findAll({
    where: {
      id: { [Op.in]: ids },
    },
  });

  return personalRecommendations;
};

export const getSimilarYachts = async (yachtId) => {
  const yacht = await Yacht.findByPk(yachtId, {
    attributes: ["similarYachts"],
  });

  if (!yacht || !yacht.similarYachts || yacht.similarYachts.length === 0) {
    return [];
  }

  const recommendationIds = yacht.similarYachts;

  const recommendedYachts = await Yacht.findAll({
    where: {
      id: { [Op.in]: recommendationIds },
    },
  });

  return recommendedYachts;
};

const triggerSimilarYachtsUpdate = () => {
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);

  dotenv.config({ path: path.resolve(__dirname, "../../.env") });

  const pythonPath = process.env.PYTHON_PATH;

  const scriptPath = path.resolve(__dirname, "../../models/similar_yachts_model.py");

  console.log("[Python] Starting script:", scriptPath);

  const pythonProcess = spawn(pythonPath, [scriptPath]);

  pythonProcess.stdout.on("data", (data) => {
    console.log(`[Python/stdout]: ${data.toString().trim()}`);
  });

  pythonProcess.stderr.on("data", (data) => {
    console.error(`[Python/stderr]: ${data.toString().trim()}`);
  });

  pythonProcess.on("close", (code) => {
    if (code === 0) {
      console.log("[Python] Script similar_yachts_model.py finished successfully ✅");
    } else {
      console.error(`[Python] Script interrupted with error code ${code} ❌`);
    }
  });
};
