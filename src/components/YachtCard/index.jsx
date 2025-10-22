import { Typography, Divider } from "@mui/material";
import DetailItem from "./DetailItem";
import {
  StyledWrapper,
  StyledImageWrapper,
  StyledImage,
  StyledIconButton,
} from "./styled";
import HeartIcon from "src/assets/icons/heart.svg";
import LocationIcon from "src/assets/icons/location.svg";
import CalendarIcon from "src/assets/icons/calendar.svg";
import ArrowsIcon from "src/assets/icons/arrows.svg";
import BedIcon from "src/assets/icons/bed.svg";
import UsersIcon from "src/assets/icons/users.svg";
import { useNavigate } from "react-router-dom";
import { ROUTES } from "src/navigation/routes";

const YachtCard = ({ yachtDetails }) => {
  const navigate = useNavigate();
  const location = useLocation();

  const onYachtClick = () => {
    navigate(replaceUrlParams(ROUTES.YACHT_DETAILS, { id: yachtDetails.id }), {
      state: location,
    });
  };
  return (
    <StyledWrapper onClick={onYachtClick}>
      <StyledImageWrapper>
        <StyledImage src={yachtDetails.thumbnail} alt={yachtDetails.name} />
        <StyledIconButton>
          <HeartIcon />
        </StyledIconButton>
      </StyledImageWrapper>
      <Stack>
        <Typography variant="subtitle">{yachtDetails.name}</Typography>
        <Stack direction={"row"}>
          <Typography>${yachtDetails.price} / day</Typography>
          <Typography color="secondary">{yachtDetails.type}</Typography>
          <Divider color="secondary" lineHeight={2} />
        </Stack>
        <Stack direction={"row"}>
          <DetailItem
            icon={CalendarIcon}
            keyTitle="Year"
            value={yachtDetails.year}
          />
          <DetailItem
            icon={ArrowsIcon}
            keyTitle="Length"
            value={yachtDetails.length}
          />
          <DetailItem
            icon={LocationIcon}
            keyTitle="Location"
            value={yachtDetails.location}
          />
          <DetailItem
            icon={BedIcon}
            keyTitle="Cabins"
            value={yachtDetails.cabins}
          />
          <DetailItem
            icon={UsersIcon}
            keyTitle="Capacity"
            value={yachtDetails.guests}
          />
        </Stack>
      </Stack>
    </StyledWrapper>
  );
};

export default YachtCard;
