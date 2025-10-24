import { useParams, useLocation } from "react-router-dom";

const YachtDetailsPage = () => {
  const { id } = useParams();
  const location = useLocation();
  const backLinkHref = location.state ?? "/";

  return (
    <div>
      <Link to={backLinkHref}>Back to yachts</Link>
      <p>Yacht Details Page {id}</p>
    </div>
  );
};

export default YachtDetailsPage;
